using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Finds the hands from the Hand Tracking building blocks, so nothing else has to
/// know where the rig lives. Cached, and re-resolved if the rig is rebuilt.
/// </summary>
public static class Hands
{
    static OVRHand[] _all;

    /// <summary>Every OVRHand in the scene (left + right building blocks).</summary>
    public static OVRHand[] All
    {
        get
        {
            if (_all == null || _all.Length == 0 || _all[0] == null)
                _all = Object.FindObjectsByType<OVRHand>();
            return _all;
        }
    }

    public static OVRHand Left => Get(OVRPlugin.Hand.HandLeft);
    public static OVRHand Right => Get(OVRPlugin.Hand.HandRight);

    public static OVRHand Get(OVRPlugin.Hand which)
    {
        foreach (var hand in All)
            if (hand != null && hand.GetHand() == which) return hand;
        return null;
    }

    public static IEnumerable<OVRHand> Tracked
    {
        get
        {
            foreach (var hand in All)
                if (hand != null && hand.IsTracked) yield return hand;
        }
    }

    /// <summary>Pointer pose of a tracked hand — where the user is aiming.</summary>
    public static bool TryGetPose(OVRPlugin.Hand which, out Pose pose)
    {
        pose = default;
        var hand = Get(which);
        if (hand == null || !hand.IsTracked) return false;
        var t = hand.PointerPose;
        pose = new Pose(t.position, t.rotation);
        return true;
    }

    public static bool IsPinching(OVRPlugin.Hand which)
    {
        var hand = Get(which);
        return hand != null && hand.IsTracked && hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
    }

    /// <summary>Index fingertip transform, or null until the skeleton has data. Matches either skeleton version.</summary>
    public static Transform IndexTip(OVRHand hand)
    {
        if (hand == null || !hand.TryGetComponent(out OVRSkeleton skeleton)) return null;
        if (!skeleton.IsDataValid || skeleton.Bones == null) return null;
        foreach (var bone in skeleton.Bones)
            if (bone.Id == OVRSkeleton.BoneId.Hand_IndexTip || bone.Id == OVRSkeleton.BoneId.XRHand_IndexTip)
                return bone.Transform;
        return null;
    }

    /// <summary>
    /// Pinch point of a hand that is pinching right now — one-line "place it here".
    /// Index fingertip from the skeleton; the pointer pose is a fallback while the skeleton warms up.
    /// </summary>
    public static bool TryGetPinchPoint(OVRPlugin.Hand which, out Vector3 point)
    {
        point = default;
        if (!IsPinching(which)) return false;
        var tip = IndexTip(Get(which));
        if (tip != null) { point = tip.position; return true; }
        if (!TryGetPose(which, out var pose)) return false;
        point = pose.position;
        return true;
    }
}
