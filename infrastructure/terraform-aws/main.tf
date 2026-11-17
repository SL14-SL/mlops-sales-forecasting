locals {
  name_prefix = "${var.project_name}-${var.environment}"
}

resource "aws_s3_bucket" "training" {
  bucket        = "${local.name_prefix}-${var.aws_account_id}-${var.aws_region}"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "training" {
  bucket = aws_s3_bucket.training.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training" {
  bucket = aws_s3_bucket.training.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "training" {
  bucket = aws_s3_bucket.training.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_ecr_repository" "training" {
  name                 = "${local.name_prefix}-training"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  encryption_configuration {
    encryption_type = "AES256"
  }

  image_scanning_configuration {
    scan_on_push = true
  }
}

# SageMaker uses one account/region-wide log group. Only the dev module owns it;
# separate staging/prod stacks must not try to create the same global name.
resource "aws_cloudwatch_log_group" "training" {
  count             = var.environment == "dev" ? 1 : 0
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 14
}

moved {
  from = aws_cloudwatch_log_group.training
  to   = aws_cloudwatch_log_group.training[0]
}
