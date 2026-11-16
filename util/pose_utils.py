import ailia

from dtype_utils import numpy_type_to_builtin_type

POSE_OBJECT_TYPES = (
    ailia.PoseEstimatorObjectPose,
    ailia.PoseEstimatorObjectUpPose,
    ailia.PoseEstimatorObjectHand,
)

def _to_json_object(value):
    """Recursively convert a namedtuple to a dict, and a list of them to a list of dicts"""
    if hasattr(value, '_fields'):
        return {
            name: _to_json_object(getattr(value, name))
            for name in value._fields
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_object(v) for v in value]

    return value


def pose_object_to_dict(obj):
    """Convert a pose estimation object to a json serializable dict

    The accepted types have different sets of fields, but no type specific
    handling is used: each is converted by walking the namedtuple _fields.

    Raises TypeError if obj is not one of the accepted types.
    """
    if not isinstance(obj, POSE_OBJECT_TYPES):
        raise TypeError(
            'obj must be one of ({}), not {}'.format(
                ', '.join(t.__name__ for t in POSE_OBJECT_TYPES),
                type(obj).__name__
            )
        )

    return numpy_type_to_builtin_type(_to_json_object(obj))
