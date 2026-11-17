resource "aws_security_group" "mlflow_link" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  name        = "${local.name_prefix}-mlflow-link"
  description = "MLflow API Gateway VPC link."
  vpc_id      = aws_vpc.training.id
}

resource "aws_security_group" "mlflow_lb" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  name        = "${local.name_prefix}-mlflow-lb"
  description = "MLflow internal ALB, reachable only from the VPC link."
  vpc_id      = aws_vpc.training.id
}

resource "aws_vpc_security_group_egress_rule" "mlflow_link" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow_link[0].id
  referenced_security_group_id = aws_security_group.mlflow_lb[0].id
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
}

resource "aws_vpc_security_group_ingress_rule" "mlflow_lb" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow_lb[0].id
  referenced_security_group_id = aws_security_group.mlflow_link[0].id
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
}

resource "aws_vpc_security_group_egress_rule" "mlflow_lb" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow_lb[0].id
  referenced_security_group_id = aws_security_group.mlflow[0].id
  ip_protocol                  = "tcp"
  from_port                    = 5000
  to_port                      = 5000
}

resource "aws_vpc_security_group_ingress_rule" "mlflow" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow[0].id
  referenced_security_group_id = aws_security_group.mlflow_lb[0].id
  ip_protocol                  = "tcp"
  from_port                    = 5000
  to_port                      = 5000
}

resource "aws_lb" "mlflow" {
  count                      = var.mlflow_managed_enabled ? 1 : 0
  name_prefix                = "mlreg-"
  internal                   = true
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.mlflow_lb[0].id]
  subnets                    = [aws_subnet.training.id, aws_subnet.api_secondary.id]
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "mlflow" {
  count                = var.mlflow_managed_enabled ? 1 : 0
  name_prefix          = "mlreg-"
  port                 = 5000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.training.id
  deregistration_delay = 30
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "mlflow" {
  count             = var.mlflow_managed_enabled ? 1 : 0
  load_balancer_arn = aws_lb.mlflow[0].arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow[0].arn
  }
}

resource "aws_apigatewayv2_vpc_link" "mlflow" {
  count              = var.mlflow_managed_enabled ? 1 : 0
  name               = "${local.name_prefix}-mlflow"
  security_group_ids = [aws_security_group.mlflow_link[0].id]
  subnet_ids         = [aws_subnet.training.id, aws_subnet.api_secondary.id]
}

resource "aws_apigatewayv2_api" "mlflow" {
  count         = var.mlflow_managed_enabled ? 1 : 0
  name          = "${local.name_prefix}-mlflow"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "mlflow" {
  count                  = var.mlflow_managed_enabled ? 1 : 0
  api_id                 = aws_apigatewayv2_api.mlflow[0].id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  integration_uri        = aws_lb_listener.mlflow[0].arn
  connection_type        = "VPC_LINK"
  connection_id          = aws_apigatewayv2_vpc_link.mlflow[0].id
  payload_format_version = "1.0"
  timeout_milliseconds   = 30000
  request_parameters     = { "overwrite:path" = "$request.path" }
}

resource "aws_cloudwatch_log_group" "mlflow_gateway" {
  count             = var.mlflow_managed_enabled ? 1 : 0
  name              = "/aws/apigateway/${local.name_prefix}-mlflow"
  retention_in_days = 14
}

resource "aws_apigatewayv2_stage" "mlflow" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  api_id      = aws_apigatewayv2_api.mlflow[0].id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = 100
    throttling_rate_limit  = 50
  }
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.mlflow_gateway[0].arn
    format          = jsonencode({ requestId = "$context.requestId", status = "$context.status", responseLength = "$context.responseLength" })
  }
}

resource "aws_cloudwatch_log_group" "mlflow_authorizer" {
  count             = var.mlflow_managed_enabled ? 1 : 0
  name              = "/aws/lambda/${local.name_prefix}-mlflow-auth"
  retention_in_days = 14
}

resource "aws_iam_role" "mlflow_authorizer" {
  count = var.mlflow_managed_enabled ? 1 : 0
  name  = "${local.name_prefix}-mlflow-auth"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "lambda.amazonaws.com" }
  }] })
}

resource "aws_iam_role_policy" "mlflow_authorizer" {
  count = var.mlflow_managed_enabled ? 1 : 0
  name  = "read-token-and-write-logs"
  role  = aws_iam_role.mlflow_authorizer[0].id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = aws_secretsmanager_secret.mlflow_token[0].arn },
    { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.mlflow_authorizer[0].arn}:*" }
  ] })
}

data "archive_file" "mlflow_authorizer" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  type        = "zip"
  source_file = "${path.module}/mlflow_authorizer/handler.py"
  output_path = "${path.module}/.terraform/mlflow-authorizer.zip"
}

resource "aws_lambda_function" "mlflow_authorizer" {
  count            = var.mlflow_managed_enabled ? 1 : 0
  function_name    = "${local.name_prefix}-mlflow-auth"
  role             = aws_iam_role.mlflow_authorizer[0].arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.mlflow_authorizer[0].output_path
  source_code_hash = data.archive_file.mlflow_authorizer[0].output_base64sha256
  timeout          = 10
  environment { variables = { TOKEN_SECRET_ARN = aws_secretsmanager_secret.mlflow_token[0].arn } }
  depends_on = [aws_iam_role_policy.mlflow_authorizer]
}

resource "aws_lambda_permission" "mlflow_authorizer" {
  count         = var.mlflow_managed_enabled ? 1 : 0
  statement_id  = "AllowMlflowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mlflow_authorizer[0].function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.mlflow[0].execution_arn}/authorizers/${aws_apigatewayv2_authorizer.mlflow[0].id}"
}

resource "aws_apigatewayv2_authorizer" "mlflow" {
  count                             = var.mlflow_managed_enabled ? 1 : 0
  api_id                            = aws_apigatewayv2_api.mlflow[0].id
  name                              = "mlflow-token"
  authorizer_type                   = "REQUEST"
  authorizer_uri                    = aws_lambda_function.mlflow_authorizer[0].invoke_arn
  authorizer_payload_format_version = "2.0"
  enable_simple_responses           = true
  authorizer_result_ttl_in_seconds  = 0
  identity_sources                  = ["$request.header.Authorization"]
}

resource "aws_apigatewayv2_route" "mlflow" {
  count              = var.mlflow_managed_enabled ? 1 : 0
  api_id             = aws_apigatewayv2_api.mlflow[0].id
  route_key          = "$default"
  target             = "integrations/${aws_apigatewayv2_integration.mlflow[0].id}"
  authorization_type = "CUSTOM"
  authorizer_id      = aws_apigatewayv2_authorizer.mlflow[0].id
  depends_on         = [aws_lambda_permission.mlflow_authorizer]
}
