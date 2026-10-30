import json
from pathlib import Path

import pytest

from experiments.fig8_extension import iso_postal_country_cache as module


def test_requires_explicit_iso_country_and_preserves_ambiguity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path=tmp_path/'iso.json';path.write_text(json.dumps({'3166-1':[
        {'alpha_2':'IN','name':'India'}, {'alpha_2':'DE','name':'Germany'},
        {'alpha_2':'JP','name':'Japan'}, {'alpha_2':'GE','name':'Georgia'},
        {'alpha_2':'CN','name':'China','official_name':"People's Republic of China"}]}))
    monkeypatch.setattr(module,'ISO',path);patterns=module.country_patterns()
    for address, code in [('Institute, City, 600036 India','IN'),('Institute, City, 12345 Germany','DE'),
                          ('Institute, City, 600-0000 Japan','JP'),('Institute, City, 100190 People’s Republic of China','CN')]:
        assert module.explicit_code(address,patterns)==code
    for address in ['Institute, City, 600036','Institute, City, 1000 Georgia',
                    'Institute, Hong Kong, 100190 China','Institute; elsewhere, City, 600036 India',
                    'Institute, City, University of India','Institute, City, 1234567 Germany']:
        assert module.explicit_code(address,patterns) is None
