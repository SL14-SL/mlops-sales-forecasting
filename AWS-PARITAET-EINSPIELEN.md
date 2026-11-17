# AWS-Parität im Forecasting-Projekt einspielen

Der Patch bezieht sich auf das hochgeladene ZIP mit Stand
`2a0c5993ea67a74442d2636f9d27033fb59200b1`. Das ZIP enthält bereits alle Änderungen.
Das Template ist nicht Bestandteil dieser Änderung.

## Patch in deinem bestehenden Repository anwenden

Im Repository `mlops-sales-forecasting` zuerst den Arbeitsstand prüfen. Eigene
Änderungen vorher sichern; bei Abweichungen vom hochgeladenen Stand keine
Patch-Konflikte mit erzwungenem Anwenden übergehen.

```bash
git status --short
git switch -c feat/aws-lifecycle-parity
git apply --check /pfad/mlops-sales-forecasting-aws-parity.patch
git apply /pfad/mlops-sales-forecasting-aws-parity.patch
uv sync --locked --extra aws
uv run --locked --extra aws ruff check .
uv run --locked --extra aws pytest -q
```

Die Prüfung umfasst echtes XGBoost-Training, eine echte lokale MLflow-Registry,
Promotion und Nicht-Promotion, Wiederherstellung nach Veröffentlichungsfehler,
portable Modellpakete und Registry-/Release-Rollback. AWS-Aufrufe für den
ECS-Betrieb sind in den lokalen Tests simuliert; das ist kein Live-Cloud-Nachweis.

## Was geändert wurde

- AWS-Fargate nutzt dieselbe vollständige Lifecycle-Logik wie GCP.
- Gemeinsame MLflow-Registry, Challenger, Champion-Vergleich auf denselben
  Validierungsdaten und die bestehenden Promotion-Guardrails sind angebunden.
- Erfolgreiche Champions werden mit Modell, Metadaten, Kalender und Zustand als
  vollständiger S3-Release veröffentlicht und vor Aktivierung geladen/geprüft.
- Laufende ECS-Tasks werden aktualisiert; die erwartete Release-ID wird geprüft.
- Modell-Rollback setzt Registry-Alias und S3-Pointer zurück und aktualisiert ECS.
- Monitoring-gesteuertes Retraining nutzt die vorhandene Policy und speichert
  einen erfolgreichen Zyklus erst nach erfolgreichem ECS-Refresh.
- AWS-Workflows für Deployment, Lifecycle, optionales geplantes Retraining und
  API-Image-Rollback sowie Remote-Terraform-State sind ergänzt.

## Vor dem ersten Cloud-Lauf

Lies `docs/aws-lifecycle.md`. Dort stehen die vollständige Funktionszuordnung,
GitHub-Variablen, OIDC-/State-Voraussetzungen und die Reihenfolge des Aufbaus.
Wie beim vorhandenen GCP-Modul wird ein externer, erreichbarer und dauerhafter
MLflow-Dienst benötigt. Diese Änderung provisioniert diesen Dienst nicht selbst.

Die früheren, fest eingebauten Smoke-Test-Image-Digests wurden entfernt.
Beide Images müssen neu gebaut werden. Der Deployment-Workflow übernimmt das.
Die alte isolierte Training-CLI bleibt als separater Smoke-Test verfügbar.

Wichtig bei vorhandenem Terraform-State: Der AWS-Stack verwendet jetzt ein
S3-Backend. Bestehenden lokalen State sichern und mit `init -migrate-state`
übertragen, statt einen leeren State über laufenden Ressourcen anzulegen.

## Verifikationsgrenzen

Ruff, **1.046 Python-Tests**, actionlint und Terraform-Formatierung wurden lokal erfolgreich geprüft.
Die Tests meldeten sieben bestehende Bibliothekswarnungen (MLflow/SQLAlchemy und Schema-Inferenz).
Die AWS-Provider-Installation funktioniert. `terraform validate` kann in der
Bearbeitungsumgebung wegen gesperrter Unix-Sockets nicht vollständig laufen;
dafür wurde die AWS-Validierung im bestehenden Terraform-CI-Workflow ergänzt.

Ein Live-Test der neuen AWS-Workflows, IAM-Rechte und gemeinsamen Remote-Registry
wurde hier nicht durchgeführt. Der nächste Schritt ist daher: Patch einspielen,
CI prüfen, gemeinsame Dienste konfigurieren und den dokumentierten Cloud-Smoke-Test
vom Bootstrap bis zur Promotion und zum Rollback durchlaufen.
