import numpy as np

from navsim.agents.rap_dino.recovery.evaluation import selected_and_best_proposal_metrics


def test_selected_and_best_proposal_metrics_reports_selection_gap():
    target = np.zeros((8, 3), dtype=np.float32)
    cv = np.ones((8, 3), dtype=np.float32)
    bad = np.ones((8, 3), dtype=np.float32) * 5.0
    good = np.zeros((8, 3), dtype=np.float32)
    prediction = {
        "trajectory": bad,
        "proposals": np.stack([bad, good], axis=0),
    }

    metrics = selected_and_best_proposal_metrics(prediction, target, cv)

    assert metrics["selected_ade"] > metrics["best_proposal_ade"]
    assert metrics["proposal_selection_ade_gap"] > 0
    assert metrics["proposal_selection_fde_gap"] > 0
