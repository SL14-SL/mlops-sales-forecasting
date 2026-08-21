from src.inference.releases.publisher import (
    publish_serving_release,
    build_release_id
)
from src.inference.releases.repository import (
    activate_release_pointer,
    load_active_release_id,
    load_active_serving_manifest,
    list_serving_release_manifests,
    load_serving_manifest, 
    load_release_prediction_probe, 
    load_serving_release_manifest,
)

from src.inference.releases.storage import (
    write_json, 
    write_text, 
    copy_uri, 
    read_text, 
    build_release_paths,
    sha256_uri,
)

from src.inference.releases.manifest import (
    parse_serving_manifest, 
    resolve_release_artifact_uri,
) 

load_release_prediction_probe = load_release_prediction_probe
load_serving_release_manifest = load_serving_release_manifest
load_serving_manifest = load_serving_manifest
publish_serving_release = publish_serving_release
activate_release_pointer = activate_release_pointer
load_active_release_id = load_active_release_id
load_active_serving_manifest = load_active_serving_manifest
list_serving_release_manifests = list_serving_release_manifests

write_text = write_text
write_json = write_json
copy_uri = copy_uri
read_text = read_text
build_release_paths = build_release_paths
sha256_uri = sha256_uri

parse_serving_manifest = parse_serving_manifest
resolve_release_artifact_uri = resolve_release_artifact_uri
build_release_id = build_release_id
