locals {
  name_prefix = "${var.project_name}-dev"
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

resource "aws_cloudwatch_log_group" "training" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 14
}