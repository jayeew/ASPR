from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig9.build_fig9_case import CASE_ID, run  # noqa: E402
from experiments.kg_perturbation_fig9.run_fig9_checkpoint_inference import (  # noqa: E402
    build_checkpoint_metadata,
    normalize_checkpoint_review_payload,
    parse_checkpoint_json,
)


class Fig9CheckpointBoundaryTests(unittest.TestCase):
    def test_checkpoint_inference_helpers_build_contract_compatible_payload(self) -> None:
        raw = (
            "Here is the review JSON:\n"
            "```json\n"
            "{\n"
            f'  "case_id": "{CASE_ID}",\n'
            '  "summary_judgement": "Novel but needs careful caveats.",\n'
            '  "major_strengths": ["Clear mechanism", "Useful revision"],\n'
            '  "major_concerns": ["Cryo-EM caveat"]\n'
            "}\n"
            "```\n"
        )
        parsed = parse_checkpoint_json(raw)
        normalized = normalize_checkpoint_review_payload(parsed, raw_text=raw)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
            (checkpoint / "model.safetensors").write_text("tiny fake weights\n", encoding="utf-8")
            metadata = build_checkpoint_metadata(
                checkpoint_path=checkpoint,
                prompt="review prompt",
                decoding_config={"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 128},
                seed=13,
                runtime_seconds=2.5,
            )

        normalized["checkpoint_metadata"] = metadata

        self.assertEqual(CASE_ID, normalized["case_id"])
        self.assertTrue(normalized["checkpoint_invoked"])
        self.assertEqual("checkpoint_generated_aspr_qwen_output", normalized["output_origin"])
        self.assertIn("Novel", normalized["summary_judgement"])
        self.assertEqual(["Clear mechanism", "Useful revision"], normalized["major_strengths"])
        self.assertEqual(["Cryo-EM caveat"], normalized["major_concerns"])
        self.assertTrue(metadata["model_hash"].startswith("sha256:"))
        self.assertEqual("review prompt", metadata["prompt"])
        self.assertEqual(13, metadata["seed"])

    def test_fig9_writes_standard_quality_report_and_keeps_checkpoint_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            markdown_root = tmp_path / "markdown"
            out_dir = tmp_path / "fig9"
            paper_dir = markdown_root / "paper"
            peer_dir = markdown_root / "peer_review"
            paper_dir.mkdir(parents=True)
            peer_dir.mkdir(parents=True)
            paper_text = "\n".join(f"paper line {idx}" for idx in range(1, 120))
            peer_text = "\n".join(f"peer line {idx}" for idx in range(1, 620))
            (paper_dir / f"{CASE_ID}.md").write_text(paper_text, encoding="utf-8")
            (peer_dir / f"{CASE_ID}_r.md").write_text(peer_text, encoding="utf-8")

            report = run(markdown_root, out_dir)
            quality = json.loads((out_dir / "figure_quality_report.json").read_text(encoding="utf-8"))
            manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            metadata_template_path = out_dir / "fig9_checkpoint_metadata_template.json"
            checkpoint_contract_path = out_dir / "fig9_checkpoint_run_contract.json"
            self.assertTrue(metadata_template_path.exists())
            self.assertTrue(checkpoint_contract_path.exists())
            metadata_template = json.loads(metadata_template_path.read_text(encoding="utf-8"))
            checkpoint_contract = json.loads(checkpoint_contract_path.read_text(encoding="utf-8"))
            publication_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in [
                    out_dir / "fig9_panel_text.json",
                    out_dir / "fig9_fig8_module_alignment.csv",
                    out_dir / "fig9_assumed_aspr_qwen_output.json",
                    out_dir / "fig9_fusion_output.json",
                    out_dir / "figure_quality_report.json",
                ]
                if path.exists()
            )

        self.assertTrue(report["complete"])
        self.assertEqual("fig9", quality["figure"])
        self.assertEqual("prototype_run_instance_checkpoint_placeholder", quality["status_label"])
        self.assertEqual(0, quality["quality_gates"]["checkpoint_generated_aspr_qwen"])
        self.assertEqual(1, quality["quality_gates"]["checks"]["large_run_instance_visual"])
        self.assertEqual(1, quality["quality_gates"]["checks"]["manifest_bound_visual"])
        self.assertEqual(1, quality["quality_gates"]["checks"]["visible_text_compacted"])
        self.assertEqual(1, quality["quality_gates"]["checks"]["main_visual_panel_count_le_3"])
        self.assertEqual(1, quality["quality_gates"]["checks"]["gpt_image_visual_layer_contract"])
        self.assertIn("assumed pipeline-ready placeholder", quality["aspr_qwen_boundary"])
        self.assertEqual("fig9", manifest["figure"])
        self.assertEqual("fig9_checkpoint_metadata.json", checkpoint_contract["checkpoint_metadata_path"])
        self.assertEqual("fig9_aspr_qwen_output.json", checkpoint_contract["checkpoint_output_path"])
        self.assertTrue(set(checkpoint_contract["required_metadata_keys"]).issubset(metadata_template))
        self.assertEqual("sha256:<model-or-adapter-hash>", metadata_template["model_hash"])
        self.assertNotIn("human-like", publication_text.lower())

    def test_fig9_preserves_checkpoint_generated_qwen_output_when_metadata_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            markdown_root = tmp_path / "markdown"
            out_dir = tmp_path / "fig9"
            paper_dir = markdown_root / "paper"
            peer_dir = markdown_root / "peer_review"
            paper_dir.mkdir(parents=True)
            peer_dir.mkdir(parents=True)
            (paper_dir / f"{CASE_ID}.md").write_text("\n".join(f"paper line {idx}" for idx in range(1, 120)), encoding="utf-8")
            (peer_dir / f"{CASE_ID}_r.md").write_text("\n".join(f"peer line {idx}" for idx in range(1, 620)), encoding="utf-8")
            out_dir.mkdir(parents=True)
            checkpoint_output = {
                "case_id": CASE_ID,
                "output_origin": "checkpoint_generated_aspr_qwen_output",
                "checkpoint_invoked": True,
                "pipeline_ready": True,
                "summary_judgement": "Checkpoint-generated review summary.",
                "major_strengths": ["strength"],
                "major_concerns": ["concern"],
                "checkpoint_metadata": {
                    "model_hash": "sha256:abc",
                    "training_config": {"epochs": 1},
                    "data_version": "v1",
                    "prompt": "review this manuscript",
                    "decoding_config": {"temperature": 0.1},
                    "seed": 7,
                    "runtime_seconds": 12.3,
                },
            }
            (out_dir / "fig9_aspr_qwen_output.json").write_text(
                json.dumps(checkpoint_output, indent=2) + "\n",
                encoding="utf-8",
            )

            run(markdown_root, out_dir)
            qwen = json.loads((out_dir / "fig9_aspr_qwen_output.json").read_text(encoding="utf-8"))
            quality = json.loads((out_dir / "figure_quality_report.json").read_text(encoding="utf-8"))
            fusion = json.loads((out_dir / "fig9_fusion_output.json").read_text(encoding="utf-8"))

        self.assertTrue(qwen["checkpoint_invoked"])
        self.assertEqual("sha256:abc", qwen["checkpoint_metadata"]["model_hash"])
        self.assertEqual("checkpoint_case_run_instance_ready", quality["status_label"])
        self.assertEqual(1, quality["quality_gates"]["checkpoint_generated_aspr_qwen"])
        self.assertEqual(1, quality["quality_gates"]["main_claim_ready"])
        self.assertFalse(fusion["provenance"]["aspr_qwen_assumed"])

    def test_fig9_merges_checkpoint_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            markdown_root = tmp_path / "markdown"
            out_dir = tmp_path / "fig9"
            paper_dir = markdown_root / "paper"
            peer_dir = markdown_root / "peer_review"
            paper_dir.mkdir(parents=True)
            peer_dir.mkdir(parents=True)
            (paper_dir / f"{CASE_ID}.md").write_text("\n".join(f"paper line {idx}" for idx in range(1, 120)), encoding="utf-8")
            (peer_dir / f"{CASE_ID}_r.md").write_text("\n".join(f"peer line {idx}" for idx in range(1, 620)), encoding="utf-8")
            out_dir.mkdir(parents=True)
            checkpoint_output = {
                "case_id": CASE_ID,
                "output_origin": "checkpoint_generated_aspr_qwen_output",
                "checkpoint_invoked": True,
                "pipeline_ready": True,
                "summary_judgement": "Checkpoint-generated review summary.",
                "major_strengths": ["strength"],
                "major_concerns": ["concern"],
            }
            checkpoint_metadata = {
                "model_hash": "sha256:sidecar",
                "training_config": {"epochs": 2},
                "data_version": "v2",
                "prompt": "review this manuscript with evidence",
                "decoding_config": {"temperature": 0.2},
                "seed": 11,
                "runtime_seconds": 19.5,
            }
            (out_dir / "fig9_aspr_qwen_output.json").write_text(
                json.dumps(checkpoint_output, indent=2) + "\n",
                encoding="utf-8",
            )
            (out_dir / "fig9_checkpoint_metadata.json").write_text(
                json.dumps(checkpoint_metadata, indent=2) + "\n",
                encoding="utf-8",
            )

            run(markdown_root, out_dir)
            qwen = json.loads((out_dir / "fig9_aspr_qwen_output.json").read_text(encoding="utf-8"))
            quality = json.loads((out_dir / "figure_quality_report.json").read_text(encoding="utf-8"))

        self.assertEqual("sha256:sidecar", qwen["checkpoint_metadata"]["model_hash"])
        self.assertEqual("sha256:sidecar", qwen["model_hash"])
        self.assertEqual("checkpoint_case_run_instance_ready", quality["status_label"])
        self.assertEqual(1, quality["quality_gates"]["checkpoint_generated_aspr_qwen"])

    def test_fig9_rejects_checkpoint_metadata_without_required_review_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            markdown_root = tmp_path / "markdown"
            out_dir = tmp_path / "fig9"
            paper_dir = markdown_root / "paper"
            peer_dir = markdown_root / "peer_review"
            paper_dir.mkdir(parents=True)
            peer_dir.mkdir(parents=True)
            (paper_dir / f"{CASE_ID}.md").write_text("\n".join(f"paper line {idx}" for idx in range(1, 120)), encoding="utf-8")
            (peer_dir / f"{CASE_ID}_r.md").write_text("\n".join(f"peer line {idx}" for idx in range(1, 620)), encoding="utf-8")
            out_dir.mkdir(parents=True)
            checkpoint_output = {
                "case_id": CASE_ID,
                "output_origin": "checkpoint_generated_aspr_qwen_output",
                "checkpoint_invoked": True,
                "pipeline_ready": True,
                "checkpoint_metadata": {
                    "model_hash": "sha256:metadata-only",
                    "training_config": {"epochs": 1},
                    "data_version": "v1",
                    "prompt": "review this manuscript",
                    "decoding_config": {"temperature": 0.1},
                    "seed": 7,
                    "runtime_seconds": 12.3,
                },
            }
            (out_dir / "fig9_aspr_qwen_output.json").write_text(
                json.dumps(checkpoint_output, indent=2) + "\n",
                encoding="utf-8",
            )

            run(markdown_root, out_dir)
            qwen = json.loads((out_dir / "fig9_aspr_qwen_output.json").read_text(encoding="utf-8"))
            quality = json.loads((out_dir / "figure_quality_report.json").read_text(encoding="utf-8"))

        self.assertFalse(qwen["checkpoint_invoked"])
        self.assertEqual("prototype_run_instance_checkpoint_placeholder", quality["status_label"])
        self.assertEqual(0, quality["quality_gates"]["checkpoint_generated_aspr_qwen"])


if __name__ == "__main__":
    unittest.main()
