"""Focused financial-rule coverage for the red-flag validator."""

from tests.validators.red_flag_validator_cases import (
    TestCyclicalPeakDetection,
    TestOCFNIRatioCheck,
    TestRealWorldEdgeCases,
    TestRealWorldSectorExamples,
    TestSectorAwareRedFlags,
    TestStrictDetectRedFlags,
    TestThinConsensusFlag,
    TestUnreliablePEG,
    TestUnreliablePEGHighGrowth,
    TestUnsustainableDistribution,
)

__all__ = [
    "TestRealWorldEdgeCases",
    "TestSectorAwareRedFlags",
    "TestRealWorldSectorExamples",
    "TestUnsustainableDistribution",
    "TestCyclicalPeakDetection",
    "TestOCFNIRatioCheck",
    "TestUnreliablePEG",
    "TestThinConsensusFlag",
    "TestStrictDetectRedFlags",
    "TestUnreliablePEGHighGrowth",
]
