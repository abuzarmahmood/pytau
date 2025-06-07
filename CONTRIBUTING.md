# Contributing to PyTau

Thank you for considering contributing to PyTau! Here are some guidelines to help you get started:

## Code Style
- Follow PEP 8 guidelines for Python code. You can refer to the [PEP 8 documentation](https://www.python.org/dev/peps/pep-0008/) for more details.
- Example of PEP 8 compliance:
  ```python
  def example_function(param1, param2):
      """This is an example function docstring."""
      # This is a comment explaining the next line of code
      result = param1 + param2
      return result
  ```
- Ensure code is well-documented with comments and docstrings.

## Pull Request Process
1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bugfix: `git checkout -b feature-name`.
3. Make your changes and commit them with clear, concise messages: `git commit -m "Description of changes"`.
4. Sync your branch with the upstream repository to avoid conflicts: `git fetch upstream` and `git merge upstream/main`.
5. Push your branch to your fork: `git push origin feature-name`.
6. Submit a pull request with a description of your changes and link to any related issues.

## Reporting Issues
- Use the issue tracker to report bugs or request features.
- Please include the following in your issue report:
  - A clear and descriptive title.
  - Steps to reproduce the issue.
  - Expected and actual results.
  - Any relevant logs or screenshots.

## Testing
- Ensure that all tests pass before submitting a pull request. You can run tests using `pytest` by executing `pytest tests/` in the terminal.
- Add new tests for any new features or bug fixes. Place your test files in the `tests/` directory and follow the existing test structure.

We appreciate your contributions and look forward to working with you!
## Code of Conduct
- We are committed to providing a friendly, safe, and welcoming environment for all. Please read and adhere to our [Code of Conduct](link-to-code-of-conduct).

## Contribution Workflow
1. Check for open issues or start a discussion to ensure your idea aligns with the project goals.
2. Follow the code style and testing guidelines.
3. Make sure your code is well-documented.
4. Submit your pull request and engage in the review process.
