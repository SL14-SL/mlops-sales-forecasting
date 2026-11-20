data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.aws_account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values = [
        "arn:aws:sagemaker:${var.aws_region}:${var.aws_account_id}:*"
      ]
    }
  }
}

resource "aws_iam_role" "training" {
  name               = "${local.name_prefix}-training"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "training" {
  statement {
    sid = "ReadBucketLocation"

    actions   = ["s3:GetBucketLocation"]
    resources = [aws_s3_bucket.training.arn]
  }

  statement {
    sid = "ListTrainingInputs"

    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.training.arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["training/input", "training/input/*"]
    }
  }

  statement {
    sid = "ReadTrainingInputs"

    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.training.arn}/training/input/*"
    ]
  }

  statement {
    sid = "WriteTrainingOutputs"

    actions = [
      "s3:PutObject",
      "s3:AbortMultipartUpload"
    ]

    resources = [
      "${aws_s3_bucket.training.arn}/training/output/*"
    ]
  }

  statement {
    sid = "AuthenticateToEcr"

    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullTrainingImage"

    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer"
    ]

    resources = [aws_ecr_repository.training.arn]
  }

  statement {
    sid = "WriteTrainingLogs"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents"
    ]

    resources = [
      "arn:aws:logs:${var.aws_region}:${var.aws_account_id}:log-group:/aws/sagemaker/TrainingJobs:*"
    ]
  }
}

resource "aws_iam_role_policy" "training" {
  name   = "training-resources"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}