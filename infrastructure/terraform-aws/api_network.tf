resource "aws_subnet" "api_secondary" {
  vpc_id                  = aws_vpc.training.id
  cidr_block              = "10.42.2.0/24"
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = false

  tags = {
    Name = "${local.name_prefix}-api-secondary"
  }
}

resource "aws_route_table_association" "api_secondary" {
  subnet_id      = aws_subnet.api_secondary.id
  route_table_id = aws_route_table.training.id
}

resource "aws_security_group" "api_link" {
  name        = "${local.name_prefix}-api-link"
  description = "API Gateway VPC link; outbound access to the internal ALB."
  vpc_id      = aws_vpc.training.id
}

resource "aws_security_group" "api_lb" {
  name        = "${local.name_prefix}-api-lb"
  description = "Internal ALB; accepts traffic only from the API Gateway VPC link."
  vpc_id      = aws_vpc.training.id
}

resource "aws_security_group" "api_task" {
  name        = "${local.name_prefix}-api"
  description = "API task; accepts port 8000 only from the internal ALB."
  vpc_id      = aws_vpc.training.id
}

resource "aws_vpc_security_group_egress_rule" "api_link_to_lb" {
  security_group_id            = aws_security_group.api_link.id
  referenced_security_group_id = aws_security_group.api_lb.id
  description                  = "Forward API Gateway requests to the internal ALB."
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
}

resource "aws_vpc_security_group_ingress_rule" "api_lb_from_link" {
  security_group_id            = aws_security_group.api_lb.id
  referenced_security_group_id = aws_security_group.api_link.id
  description                  = "Accept requests from the API Gateway VPC link."
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
}

resource "aws_vpc_security_group_egress_rule" "api_lb_to_task" {
  security_group_id            = aws_security_group.api_lb.id
  referenced_security_group_id = aws_security_group.api_task.id
  description                  = "Forward requests and health checks to the API."
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
}

resource "aws_vpc_security_group_ingress_rule" "api_task_from_lb" {
  security_group_id            = aws_security_group.api_task.id
  referenced_security_group_id = aws_security_group.api_lb.id
  description                  = "Accept requests and health checks from the internal ALB."
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
}

# Development API accesses S3, ECR, Secrets Manager and CloudWatch via public endpoints.
# Only outbound TCP/443 is allowed, but destination IPs are unrestricted.
# Reassess for production or confidential workloads.
# trivy:ignore:AVD-AWS-0104
resource "aws_vpc_security_group_egress_rule" "api_task_https" {
  security_group_id = aws_security_group.api_task.id
  description       = "HTTPS access to AWS service endpoints."
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}