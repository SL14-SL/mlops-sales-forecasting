resource "aws_apigatewayv2_vpc_link" "api" {
  name               = "${local.name_prefix}-api"
  security_group_ids = [aws_security_group.api_link.id]

  subnet_ids = [
    aws_subnet.training.id,
    aws_subnet.api_secondary.id
  ]
}

resource "aws_apigatewayv2_api" "api" {
  name          = "${local.name_prefix}-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "api" {
  api_id = aws_apigatewayv2_api.api.id

  integration_type   = "HTTP_PROXY"
  integration_method = "ANY"
  integration_uri    = aws_lb_listener.api.arn

  connection_type = "VPC_LINK"
  connection_id   = aws_apigatewayv2_vpc_link.api.id

  payload_format_version = "1.0"
  timeout_milliseconds   = 30000

  request_parameters = {
    "overwrite:path" = "$request.path"
  }
}

resource "aws_apigatewayv2_route" "api" {
  for_each = toset([
    "GET /livez",
    "GET /readyz",
    "POST /predict",
    "GET /docs",
    "GET /openapi.json"
  ])

  api_id    = aws_apigatewayv2_api.api.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${local.name_prefix}-api"
  retention_in_days = 14
}

resource "aws_apigatewayv2_stage" "api" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 20
    throttling_rate_limit  = 10
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn

    format = jsonencode({
      requestId        = "$context.requestId"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}