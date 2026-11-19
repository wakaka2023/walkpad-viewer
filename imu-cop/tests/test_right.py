import pytest
import torch
from torch import nn

from imu_cop import RightFootCoP
from imu_cop.contact_gate import ContactGate


class FixedCoordinateModel(nn.Module):
    def forward(self, imu, mask):
        return imu.new_full((len(imu), 2, 2), .22), imu.new_full((len(imu), 2), 2), None


class FixedShortGate(nn.Module):
    def forward(self, imu, mask):
        return imu.new_full((len(imu), 2), -2)


def test_right_gate_changes_only_visibility_and_requires_available_endpoint():
    model = RightFootCoP(FixedCoordinateModel(), FixedShortGate(), dict(threshold=.7, new_logit_weight=.65))
    window = torch.zeros(2, 2, 6, 150)
    mask = torch.ones(2, 2, 150, dtype=torch.bool)
    mask[1, 0, -1] = False
    cop, score, show = model(window, mask)
    torch.testing.assert_close(cop, torch.full_like(cop, .22))
    torch.testing.assert_close(score[:, 0], torch.tensor(2.).sigmoid().expand(2))
    assert show.tolist() == [[True, False], [False, False]]


def test_short_gate_uses_only_recent_available_samples():
    gate = ContactGate(relative=True).eval()
    imu = torch.randn(1, 2, 6, 150)
    mask = torch.ones(1, 2, 150)
    mask[:, 1] = 0
    with torch.inference_mode():
        predicted = gate(imu, mask)[0, 0]
        imu[..., :-31] = 1e4
        imu[:, 1] = 1e4
        torch.testing.assert_close(gate(imu, mask)[0, 0], predicted)


def test_unexpected_weights_are_rejected_before_deserialization(tmp_path):
    invalid = tmp_path / 'wrong.pt'
    invalid.write_bytes(b'unexpected weight data')
    with pytest.raises(ValueError, match='checksum'):
        RightFootCoP.load(invalid, invalid)
