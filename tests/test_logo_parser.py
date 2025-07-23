def test_logo_parser_basic():
    from src.data_pipeline.logo_parser import LogoParser
    parser = LogoParser()
    vertices = parser.parse_logo_script('tests/test.logo')
    assert isinstance(vertices, list)
    assert len(vertices) > 0
