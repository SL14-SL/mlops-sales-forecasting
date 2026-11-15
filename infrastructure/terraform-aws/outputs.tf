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