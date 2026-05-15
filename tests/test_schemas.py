# tests/test_schemas.py
from src.schemas import SVOTriplet, SVOList

def test_relation_normalization():
    # Valid canonical
    t = SVOTriplet(subject="Obama", relation="LOCATED_IN", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Alias normalization
    t = SVOTriplet(subject="Obama", relation="is in", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Case-insensitive
    t = SVOTriplet(subject="Obama", relation="located_in", object="Berlin")
    assert t.relation == "LOCATED_IN"

    # Unknown → HAS_STATE (not crash)
    t = SVOTriplet(subject="Obama", relation="XYZ_UNKNOWN", object="something")
    assert t.relation == "HAS_STATE"

    print("✅ All schema tests passed")

test_relation_normalization()