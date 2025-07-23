def test_logo_parser_basic():
    from src.data_pipeline.logo_parser import LogoParser
    from src.data_pipeline.bongard_loader import BongardLoader
    loader = BongardLoader()
    problems = loader.load_problems()
    selected_problems = loader.select_random_subset(problems, n_select=5)
    parser = LogoParser()
    vertices = parser.parse_logo_script('tests/test.logo')
    assert isinstance(vertices, list)
    assert len(vertices) > 0
