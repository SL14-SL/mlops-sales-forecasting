output "aws_region" {
  value = var.aws_region
}

output "training_bucket_name" {
  value = aws_s3_bucket.training.id
}

output "training_input_uri" {
  value = "s3://${aws_s3_bucket.training.id}/training/input"
}

output "training_output_uri" {
  value = "s3://${aws_s3_bucket.training.id}/training/output"
}

output "training_repository_url" {
  value = aws_ecr_repository.training.repository_url
}

output "training_role_arn" {
  value = aws_iam_role.training.arn
}

output "training_vpc_id" {
  value = aws_vpc.training.id
}

output "training_subnet_id" {
  value = aws_subnet.training.id
}

output "training_security_group_id" {
  value = aws_security_group.training.id
}

output "fargate_execution_role_arn" {
  value = aws_iam_role.fargate_execution.arn
}

output "fargate_task_role_arn" {
  value = aws_iam_role.fargate_task.arn
}

output "fargate_training_log_group" {
  value = aws_cloudwatch_log_group.fargate_training.name
}

output "fargate_training_cluster_arn" {
  value = aws_ecs_cluster.training.arn
}

output "fargate_training_task_definition_arn" {
  value = aws_ecs_task_definition.training.arn
}

output "fargate_training_image_uri" {
  value = "${aws_ecr_repository.training.repository_url}@${var.fargate_training_image_digest}"
}

output "api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_execution_role_arn" {
  value = aws_iam_role.api_execution.arn
}

output "api_task_role_arn" {
  value = aws_iam_role.api_task.arn
}

output "api_key_secret_arn" {
  value = aws_secretsmanager_secret.api_key.arn
}

output "api_log_group" {
  value = aws_cloudwatch_log_group.api.name
}

output "api_task_definition_arn" {
  value = aws_ecs_task_definition.api.arn
}

output "api_endpoint" {
  value = var.api_serving_enabled ? aws_apigatewayv2_api.api[0].api_endpoint : null
}

output "api_service_name" {
  value = var.api_serving_enabled ? aws_ecs_service.api[0].name : null
}

output "api_gateway_log_group" {
  value = aws_cloudwatch_log_group.api_gateway.name
}

output "lifecycle_raw_data_uri" {
  value = "s3://${aws_s3_bucket.training.id}/data/raw"
}

output "serving_models_uri" {
  value = "s3://${aws_s3_bucket.training.id}/models"
}

output "api_image_uri" {
  value = "${aws_ecr_repository.api.repository_url}@${var.api_image_digest}"
}


output "mlflow_managed_enabled" {
  value = var.mlflow_managed_enabled
}
output "mlflow_tracking_uri" {
  value = local.mlflow_tracking_uri
}
output "mlflow_token_secret_arn" {
  value = local.mlflow_token_secret_arn
}
output "mlflow_repository_url" {
  value = var.mlflow_managed_enabled ? aws_ecr_repository.mlflow[0].repository_url : null
}
output "mlflow_image_uri" {
  value = var.mlflow_managed_enabled ? "${aws_ecr_repository.mlflow[0].repository_url}@${var.mlflow_image_digest}" : null
}
output "mlflow_service_name" {
  value = var.mlflow_managed_enabled ? aws_ecs_service.mlflow[0].name : null
}
output "mlflow_database_identifier" {
  value = var.mlflow_managed_enabled ? aws_db_instance.mlflow[0].identifier : null
}
