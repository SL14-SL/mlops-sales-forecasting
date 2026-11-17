# MLflow auf AWS ergänzen

Dieser Patch baut auf dem Stand aus PR #73 (`feat/aws-lifecycle-parity`) auf.
Er ergänzt die Bereitstellung des zuvor noch vorausgesetzten MLflow-Dienstes.
Der bisherige AWS-Paritäts-Patch darf nicht erneut angewendet werden.

## 1. Änderungen einspielen und prüfen

Lege `mlops-sales-forecasting-aws-mlflow.patch` neben den Projektordner.
Im Repository bei sauberem Arbeitsstand:

```bash
git switch main
git pull --ff-only
git switch -c feat/aws-managed-mlflow
git apply --check ../mlops-sales-forecasting-aws-mlflow.patch
git apply ../mlops-sales-forecasting-aws-mlflow.patch
uv sync --locked --extra aws --extra mlflow-server
uv run --locked --extra aws --extra mlflow-server ruff check .
uv run --locked --extra aws --extra mlflow-server pytest -q
terraform -chdir=infrastructure/terraform-aws init -input=false
terraform -chdir=infrastructure/terraform-aws fmt -check -recursive
terraform -chdir=infrastructure/terraform-aws validate
```

`init` verwendet das bereits konfigurierte S3-Backend und installiert zusätzlich
den Archive-Provider. Keine erneute Backend-Migration nötig.
Bei einem Patch-Konflikt oder Prüfungsfehler stoppen und die Ausgabe prüfen.

Nach erfolgreicher Prüfung committen, pushen und einen PR erstellen. Die
bestehende Terraform-CI prüft das erweiterte Modul; die Security-CI baut und
scannt zusätzlich das MLflow-Image. Erst nach erfolgreicher CI mergen.

## 2. Deployment-Rolle erweitern

Lade die aktualisierte `aws-dev-deploy-policy.json` herunter und ersetze die
vorherige Datei neben dem Projektordner. Nicht versehentlich die alte Version
oder einen Download mit angehängtem `(1)` verwenden.

```bash
aws accessanalyzer validate-policy \
  --policy-document file://../aws-dev-deploy-policy.json \
  --policy-type IDENTITY_POLICY --region eu-north-1 \
  --query 'findings[?findingType==`ERROR`]' --output json
```

Nur wenn keine Fehler gemeldet werden (`[]`):

```bash
aws iam put-role-policy \
  --role-name mlops-sales-forecasting-dev-github-deploy \
  --policy-name ForecastingDevDeployment \
  --policy-document file://../aws-dev-deploy-policy.json
```

Die zusätzliche Richtlinie erlaubt die MLflow-Ressourcen (RDS, Lambda und
Anwendungsrollen) sowie die Initialisierung des MLflow-Tokens. Die bereits
bestehende OIDC-Vertrauensbeziehung bleibt unverändert.

## 3. Managed MLflow auswählen

```bash
gh variable set AWS_MANAGED_MLFLOW --env dev --body true
```

Für spätere lokale Terraform-Pläne zusätzlich in der bestehenden
`infrastructure/terraform-aws/terraform.tfvars` eintragen:

```hcl
mlflow_managed_enabled = true
```

Eine manuell gesetzte `MLFLOW_TRACKING_URI` ist dann nicht erforderlich. Terraform
verbindet den Trainingsjob mit dem bereitgestellten Dienst und Token-Secret.
Passwörter und Tokens nicht in `terraform.tfvars` schreiben.

## 4. Zunächst nur planen

Nach Merge auf `main`:

```bash
gh workflow run deploy-aws.yml --ref main \
  -f environment=dev -f enable_serving=false -f apply_changes=false
```

Den Lauf mit `gh run list --workflow deploy-aws.yml --limit 5` finden und mit
`gh run watch RUN_ID --exit-status` verfolgen. Danach den Plan in den Logs prüfen.
Noch keine kostenpflichtige Infrastruktur wird bei diesem Plan-Lauf angelegt.

Den Live-Aufbau starten wir nach erfolgreicher Plan-Prüfung. Die vollständigen
Befehle, der Persistenztest, der Zugriff auf die MLflow-Oberfläche und das
kontrollierte Abschalten stehen in `docs/aws-mlflow.md`.

## Prüfstatus

1.067 Tests, Ruff und Workflow-Syntax sind lokal erfolgreich geprüft.
Das Dependency-Audit findet unter der bestehenden MLflow-Ausnahme keine bekannten
Schwachstellen. Die vollständige Terraform-Validierung und das MLflow-Container-
Build müssen in CI beziehungsweise lokal laufen; hier blockiert die Umgebung den
Provider-Start. Ein AWS-Live-Test wurde noch nicht durchgeführt.
