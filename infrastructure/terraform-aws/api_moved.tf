moved {
  from = aws_ecs_service.api
  to   = aws_ecs_service.api[0]
}

moved {
  from = aws_lb.api
  to   = aws_lb.api[0]
}

moved {
  from = aws_lb_target_group.api
  to   = aws_lb_target_group.api[0]
}

moved {
  from = aws_lb_listener.api
  to   = aws_lb_listener.api[0]
}

moved {
  from = aws_apigatewayv2_vpc_link.api
  to   = aws_apigatewayv2_vpc_link.api[0]
}

moved {
  from = aws_apigatewayv2_api.api
  to   = aws_apigatewayv2_api.api[0]
}

moved {
  from = aws_apigatewayv2_integration.api
  to   = aws_apigatewayv2_integration.api[0]
}

moved {
  from = aws_apigatewayv2_stage.api
  to   = aws_apigatewayv2_stage.api[0]
}