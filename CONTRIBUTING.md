# Contributing to Heart Disease Risk Assessment Tool

Thank you for your interest in contributing to our project! This document provides guidelines and steps for contributing.

## 🎯 How to Contribute

1. **Fork the Repository**
   - Click the 'Fork' button on the top right of the repository page
   - Clone your forked repository to your local machine

2. **Set Up Development Environment**
   ```bash
   git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
   cd heart-disease-prediction
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a New Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the code style guidelines
   - Write clear commit messages
   - Add tests for new features
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   pytest tests/
   ```

6. **Submit a Pull Request**
   - Push your changes to your fork
   - Create a pull request to the main repository
   - Fill out the pull request template

## 📝 Code Style Guidelines

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add comments for complex logic

## 🧪 Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Maintain or improve test coverage
- Test edge cases and error conditions

## 📚 Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update comments for modified code
- Keep the documentation up-to-date

## 🐛 Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

## 💡 Feature Requests

1. Check if the feature has already been requested
2. Create a new issue with:
   - Detailed description
   - Use cases
   - Potential implementation ideas

## 🤝 Code Review Process

1. Pull requests will be reviewed by maintainers
2. Address any feedback or requested changes
3. Ensure all checks pass
4. Get at least one approval before merging

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

Thank you for contributing to the Heart Disease Risk Assessment Tool! 🚀 