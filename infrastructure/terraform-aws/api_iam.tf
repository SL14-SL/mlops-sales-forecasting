resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name_prefix}-api"
  retention_in_days = 14
}

resource "aws_secretsmanager_secret" "api_key" {
  name                    = "${local.name_prefix}/api-key"
  description             = "API authentication key; value managed outside Terraform."
  recovery_window_in_days = 7
}

resource "aws_iam_role" "api_execution" {
  name               = "${local.name_prefix}-api-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

data "aws_iam_policy_document" "api_execution" {
  statement {
    sid       = "AuthenticateToEcr"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullApiImage"

    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer"
    ]

    resources = [aws_ecr_repository.api.arn]
  }

  statement {
    sid = "WriteApiLogs"

    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]

    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }

  statement {
    sid       = "InjectApiKey"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.api_key.arn]
  }
}

resource "aws_iam_role_policy" "api_execution" {
  name   = "api-image-logs-and-secret"
  role   = aws_iam_role.api_execution.id
  policy = data.aws_iam_policy_document.api_execution.json
}

resource "aws_iam_role" "api_task" {
  name               = "${local.name_prefix}-api-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

data "aws_iam_policy_document" "api_task" {
  statement {
    sid       = "ReadBucketLocation"
    actions   = ["s3:GetBucketLocation"]
    resources = [aws_s3_bucket.training.arn]
  }

  statement {
    sid       = "ListServingArtifacts"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.training.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["models", "models/*"]
    }
  }

  statement {
    sid = "ReadServingArtifacts"

    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.training.arn}/models/*"
    ]
  }

  statement {
    sid = "WriteInferenceMonitoring"

    actions = [
      "s3:PutObject",
      "s3:AbortMultipartUpload"
    ]

    resources = [
      "${aws_s3_bucket.training.arn}/data/predictions/*"
    ]
  }
}

resource "aws_iam_role_policy" "api_task" {
  name   = "read-models-and-write-prediction-monitoring"
  role   = aws_iam_role.api_task.id
  policy = data.aws_iam_policy_document.api_task.json
}