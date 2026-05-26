from setuptools import setup, find_packages

setup(
    name="personality-insight-agent",
    version="1.0.0",
    packages=find_packages(include=["personality-insight-agent", "personality-insight-agent.*"]),
    package_dir={"personality_insight_agent": "personality-insight-agent"},
    install_requires=["pydantic>=2.0"],
)
