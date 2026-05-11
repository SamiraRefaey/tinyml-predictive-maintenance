# Contributing to TinyML Predictive Maintenance

Thank you for your interest in contributing to the TinyML Predictive Maintenance project! We welcome contributions from the community.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## How to Contribute

### 1. Reporting Issues

- Use the [GitHub Issues](https://github.com/SamiraRefaey/tinyml-predictive-maintenance/issues) page
- Provide detailed descriptions including steps to reproduce
- Include relevant error messages and system information
- Suggest potential solutions if possible

### 2. Feature Requests

- Open an issue with the "enhancement" label
- Describe the feature and its use case
- Explain why it would be valuable to the project

### 3. Code Contributions

#### Development Setup

```bash
# Clone the repository
git clone https://github.com/SamiraRefaey/tinyml-predictive-maintenance.git
cd tinyml-predictive-maintenance

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Choose an issue** or create one for the feature/bug you want to work on
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```
3. **Make your changes** following the coding standards
4. **Write tests** for new functionality
5. **Run the test suite**:
   ```bash
   pytest
   ```
6. **Format your code**:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```
7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
8. **Push and create a pull request**

### 4. Documentation

- Update docstrings for any modified functions
- Add examples for new features
- Update the README if needed
- Ensure Jupyter notebooks are updated

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Write descriptive variable and function names

### Type Hints

- Use type hints for function parameters and return values
- Import from `typing` module when needed

### Documentation

- Write docstrings for all public functions and classes
- Follow Google-style docstrings
- Include examples where appropriate

### Testing

- Write unit tests for all new functionality
- Aim for high test coverage
- Use descriptive test names
- Test edge cases and error conditions

## Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

Examples:
```
feat: add support for frequency domain features
fix: correct quantization range calculation
docs: update installation instructions
```

## Pull Request Process

1. **Ensure your PR**:
   - Has a clear title and description
   - References any related issues
   - Includes tests for new functionality
   - Passes all CI checks
   - Follows coding standards

2. **PR Review**:
   - At least one maintainer will review your PR
   - Address any feedback or requested changes
   - Once approved, a maintainer will merge your PR

3. **After Merge**:
   - Your contribution will be acknowledged
   - The change will be included in the next release

## Areas for Contribution

### High Priority
- Performance optimizations for edge devices
- Additional anomaly detection algorithms
- Real sensor data integration examples
- Comprehensive benchmarking suite

### Medium Priority
- Web dashboard for model monitoring
- REST API for model serving
- Docker containerization
- CI/CD pipeline improvements

### Low Priority
- Additional documentation and tutorials
- GUI applications
- Mobile app integration
- Cloud deployment examples

## Recognition

Contributors will be:
- Listed in the project's CONTRIBUTORS file
- Acknowledged in release notes
- Recognized for their impact on the project

## Questions?

If you have questions about contributing:
- Check existing issues and discussions
- Open a new discussion for questions
- Contact the maintainers

Thank you for contributing to TinyML Predictive Maintenance! 🚀