resource "aws_lb" "api" {
  count = var.api_serving_enabled ? 1 : 0

  name_prefix        = "mlapi-"
  internal           = true
  load_balancer_type = "application"

  security_groups = [aws_security_group.api_lb.id]

  subnets = [
    aws_subnet.training.id,
    aws_subnet.api_secondary.id
  ]

  drop_invalid_header_fields = true
  enable_deletion_protection = false
}

resource "aws_lb_target_group" "api" {
  count = var.api_serving_enabled ? 1 : 0

  name_prefix = "mlapi-"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.training.id

  deregistration_delay = 30

  health_check {
    enabled             = true
    path                = "/readyz"
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "api" {
  count = var.api_serving_enabled ? 1 : 0

  load_balancer_arn = aws_lb.api[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api[0].arn
  }
}