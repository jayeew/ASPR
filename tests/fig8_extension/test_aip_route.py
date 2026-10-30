from experiments.fig8_extension.supplement_queue import route
from experiments.fig8_extension.supplement_sources import publisher_url


def test_aip_is_not_iop_and_stale_prefix_route_uses_resolver() -> None:
    assert route('American Institute of Physics', '10.1116/example') == 'aip'
    assert route('AIP Publishing', '10.1063/5.0131663') == 'aip'
    assert route('', '10.1063/5.0131663') == 'aip'
    assert route('IOP Publishing', '10.1088/example') == 'iop'
    assert route('Institute of Physics', '10.35848/example') == 'iop'
    assert publisher_url('10.1063/5.0131663', 'iop') == 'https://doi.org/10.1063/5.0131663'
    assert publisher_url('10.1116/example', 'aip') == 'https://doi.org/10.1116/example'
