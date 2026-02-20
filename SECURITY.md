# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

The MIST-VLA team takes security issues seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: [your.email@example.com]

Include the following information:
- Type of issue (e.g., buffer overflow, code injection, etc.)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- We will acknowledge your email within 48 hours
- We will provide a detailed response within 7 days
- We will work on a fix and keep you updated on progress
- Once fixed, we will publicly disclose the vulnerability (with credit to you, if desired)

## Security Best Practices

When using MIST-VLA:

1. **Model Checkpoints**: Only load models from trusted sources
2. **Data Collection**: Validate all input data and environment configurations
3. **Environment Setup**: Use isolated environments (Docker, virtual environments)
4. **Credentials**: Never commit credentials or API keys to the repository
5. **Dependencies**: Regularly update dependencies to patch known vulnerabilities

## Known Security Considerations

### Simulation Environment
- MIST-VLA currently operates in simulation environments only
- Real robot deployment requires additional safety validation
- Collision detection is simulation-specific and may not transfer to real robots

### Model Safety
- VLA models can produce unsafe actions despite steering
- Always test in simulation before real-world deployment
- Implement emergency stop mechanisms for physical robots

### Data Privacy
- Rollout data may contain sensitive information
- Ensure proper access controls on stored data
- Consider data anonymization for shared datasets

## Responsible Disclosure

We follow a coordinated disclosure process:

1. Security issue reported privately
2. Patch developed and tested
3. New version released with security fix
4. Public disclosure of the vulnerability
5. Credit given to reporter (if desired)

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 0.1.1)
- Documented in [CHANGELOG.md](CHANGELOG.md)
- Announced in GitHub releases
- Mentioned in the README if critical

## Contact

For security concerns, contact:
- Email: your.email@example.com
- PGP Key: [Link to PGP key] (optional)

For general questions and non-security issues, please use [GitHub Issues](https://github.com/yourusername/MIST-VLA/issues).

---

Thank you for helping keep MIST-VLA and its users safe!
