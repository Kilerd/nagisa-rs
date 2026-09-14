import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch
from urllib.error import HTTPError

import release


class ReleaseTests(unittest.TestCase):
    def manifests(self, directory, main="0.2.0", model="0.1.0", dependency="=0.1.0"):
        root = Path(directory)
        (root / "model").mkdir()
        (root / "Cargo.toml").write_text(
            f'[package]\nname="ragisa"\nversion="{main}"\n'
            f'[dependencies]\nragisa-model={{path="model",version="{dependency}"}}\n')
        (root / "model/Cargo.toml").write_text(f'[package]\nname="ragisa-model"\nversion="{model}"\n')
        return root

    def test_tag_matches_main_and_model_can_keep_its_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.manifests(directory)
            self.assertEqual(release.versions(root, "v0.2.0"), {"ragisa-model": "0.1.0", "ragisa": "0.2.0"})
            for tag in ("0.2.0", "v0.1.0", "v0.2.0\n", "v0.2.0;echo wrong"):
                with self.subTest(tag=tag), self.assertRaises(ValueError):
                    release.versions(root, tag)

    def test_model_dependency_must_be_exact(self):
        for dependency in ("0.1.0", "^0.1.0", "=0.2.0"):
            with self.subTest(dependency=dependency), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    release.versions(self.manifests(directory, dependency=dependency))

    def test_prerelease_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.manifests(directory, main="0.2.0-rc.1")
            self.assertEqual(release.versions(root, "v0.2.0-rc.1")["ragisa"], "0.2.0-rc.1")

    def test_registry_errors_do_not_count_as_missing_crates(self):
        for code in (403, 429, 500):
            with self.subTest(code=code), patch("release.urllib.request.urlopen", side_effect=HTTPError("url", code, "error", {}, None)):
                with self.assertRaises(HTTPError):
                    release.registry_status("ragisa", "0.1.0")
        with patch("release.urllib.request.urlopen", side_effect=HTTPError("url", 404, "missing", {}, None)):
            self.assertEqual(release.registry_status("ragisa", "0.1.0"), {"crate_exists": False, "published": False})

    def test_published_and_yanked_versions(self):
        for yanked in (False, True):
            payload = {"crate": {"name": "ragisa"}, "versions": [{"num": "0.1.0", "yanked": yanked}]}
            with patch("release.urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
                if yanked:
                    with self.assertRaises(ValueError):
                        release.registry_status("ragisa", "0.1.0")
                else:
                    self.assertTrue(release.registry_status("ragisa", "0.1.0")["published"])

    def test_publish_is_tag_only_and_model_precedes_parent(self):
        versions = {"ragisa-model": "0.1.0", "ragisa": "0.2.0"}
        env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/tags/v0.2.0",
               "CARGO_REGISTRY_TOKEN": "unit-test-placeholder"}
        states = {name: {"version": version, "crate_exists": False, "published": False} for name, version in versions.items()}
        with patch.dict(os.environ, env, clear=True), patch("release.plan", return_value=states), patch("release.subprocess.run") as run:
            release.publish(versions, "v0.2.0")
            self.assertEqual([call.args[0][3] for call in run.call_args_list], ["ragisa-model", "ragisa"])
            run.reset_mock()
            states["ragisa-model"]["published"] = True
            release.publish(versions, "v0.2.0")
            self.assertEqual([call.args[0][3] for call in run.call_args_list], ["ragisa"])
            run.reset_mock()
            os.environ["GITHUB_EVENT_NAME"] = "workflow_dispatch"
            with self.assertRaises(ValueError):
                release.publish(versions, "v0.2.0")
            run.assert_not_called()

    def test_model_publish_failure_stops_the_parent(self):
        versions = {"ragisa-model": "0.1.0", "ragisa": "0.1.0"}
        env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/tags/v0.1.0",
               "CARGO_REGISTRY_TOKEN": "unit-test-placeholder"}
        states = {name: {"version": version, "published": False} for name, version in versions.items()}
        with patch.dict(os.environ, env, clear=True), patch("release.plan", return_value=states), patch(
            "release.subprocess.run", side_effect=subprocess.CalledProcessError(101, "cargo")
        ) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                release.publish(versions, "v0.1.0")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][3], "ragisa-model")


if __name__ == "__main__":
    unittest.main()
