# Modular Bongard Generator - Test Suite

This directory contains comprehensive tests for the modular bongard_generator package.

## Test Structure

### Unit Tests
- `test_config_loader.py` - Configuration management tests
- `test_rule_loader.py` - Rule loading and validation tests  
- `test_draw_utils.py` - Advanced drawing utilities tests
- `test_relation_sampler.py` - Spatial/topological relation tests
- `test_cp_sampler.py` - CP-SAT optimization tests
- `test_fallback_samplers.py` - Fallback sampling strategy tests
- `test_cache_stratification.py` - Cache and stratification tests

### Integration Tests
- `test_integration.py` - End-to-end package integration tests

### Test Configuration
- `conftest.py` - pytest configuration and fixtures
- `run_tests.py` - Test runner with coverage support

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Quick Tests (exclude slow tests)
```bash
python tests/run_tests.py --quick
```

### Run Specific Test Module
```bash
python tests/run_tests.py --test config_loader
```

### Run with pytest directly
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=src.bongard_generator --cov-report=html
```

## Test Categories

### Unit Tests (`-m unit`)
Fast, isolated tests for individual components.

### Integration Tests (`-m integration`) 
Tests that verify component interactions.

### Slow Tests (`-m slow`)
Performance and stress tests. Excluded by default in quick mode.

## Test Requirements

Required packages for running tests:
- `pytest`
- `pytest-cov` (for coverage)
- `pytest-timeout` (for test timeouts)

Install with:
```bash
pip install pytest pytest-cov pytest-timeout
```

## Mock Dependencies

Tests use mocking for external dependencies:
- OR-Tools CP-SAT solver (for unit tests)
- PIL/Pillow operations (when needed)
- File system operations

## Coverage Goals

Target coverage levels:
- Unit tests: >90% line coverage
- Integration tests: >80% functional coverage
- Critical paths: 100% coverage

## Performance Benchmarks

Slow tests include performance benchmarks:
- Generation speed tests
- Memory usage validation
- Scalability verification

Run performance tests with:
```bash
pytest tests/ -m slow -v
```

## Test Data

Tests use minimal synthetic data and mocked components to ensure:
- Fast execution
- Deterministic results
- No external dependencies
- Isolated component testing

## Debugging Tests

For debugging failed tests:
```bash
pytest tests/test_name.py::test_function -v -s --tb=long
```

For interactive debugging:
```bash
pytest tests/test_name.py::test_function --pdb
```
