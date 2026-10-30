from experiments.fig8_extension.fig8b_qualification_decide import type_class


def test_native_explicit_original_types_only() -> None:
    assert type_class('Original Papers','publisher_jstage_html')=='research'
    assert type_class('Original Paper','publisher_jstage_pdf')=='research'
    assert type_class('Article','publisher_jstage_html')=='unresolved_type'
    assert type_class('Regular Article','publisher_jstage_html')=='unresolved_type'
    assert type_class('Original Papers','publisher_oai_dc')=='unresolved_type'


def test_native_case_and_protocol_categories_keep_the_hold() -> None:
    for label in ['Case Report','Study Protocol','Data Paper']:
        assert type_class('Original Papers','publisher_jstage_html',[label])=='subtype_review_required'
