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