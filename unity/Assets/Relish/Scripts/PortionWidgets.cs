using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// One PortionWidget per tracked object. Feed it a frame of detections and it spawns,
/// smooths, ages out and retires; the prefab stays dumb. Perception reports at ~1 fps,
/// so positions are lerped between reports instead of snapped.
/// </summary>
public class PortionWidgets : MonoBehaviour
{
    /// <summary>The contract with perception. Fill this from whatever the transport delivers.</summary>
    public struct Detection
    {
        public int Id;            // tracker obj_id
        public Vector3 Position;  // world, metres
        public float Diameter;    // metres
        public float Score;       // 0..1
    }

    [SerializeField] PortionWidget prefab;
    [SerializeField, Range(0f, 1f)] float minScore = 0.3f;
    [SerializeField] float smoothing = 8f;      // per second; higher snaps faster
    [SerializeField] float staleAfter = 1.5f;   // seconds unseen before dimming
    [SerializeField] float retireAfter = 4f;    // seconds unseen before removal
    [SerializeField] bool debugLabels = true;

    class Slot { public PortionWidget Widget; public Vector3 Target; public float LastSeen; }

    readonly Dictionary<int, Slot> _slots = new();
    readonly List<int> _retired = new();

    public int Count => _slots.Count;

    /// <summary>Report one frame of detections. Ids missing from later frames age out on their own.</summary>
    public void Report(IEnumerable<Detection> frame)
    {
        foreach (var d in frame)
        {
            if (d.Score < minScore) continue;
            if (!_slots.TryGetValue(d.Id, out var slot))
            {
                slot = new Slot { Widget = Instantiate(prefab, d.Position, Quaternion.identity, transform) };
                slot.Widget.name = $"PortionWidget {d.Id}";
                _slots[d.Id] = slot;
            }
            slot.Target = d.Position;
            slot.LastSeen = Time.time;
            slot.Widget.SetDiameter(d.Diameter);
            slot.Widget.SetStale(false);
            slot.Widget.SetLabel(debugLabels ? $"#{d.Id}  {d.Diameter * 100f:0} cm  {d.Score:P0}" : null);
        }
    }

    public void Success(int id)
    {
        if (_slots.TryGetValue(id, out var slot)) slot.Widget.Success();
    }

    public void Clear()
    {
        foreach (var slot in _slots.Values) Destroy(slot.Widget.gameObject);
        _slots.Clear();
    }

    void Update()
    {
        float k = 1f - Mathf.Exp(-smoothing * Time.deltaTime);   // frame-rate independent lerp
        foreach (var kv in _slots)
        {
            var slot = kv.Value;
            float unseen = Time.time - slot.LastSeen;
            if (unseen > retireAfter) { _retired.Add(kv.Key); continue; }
            slot.Widget.SetStale(unseen > staleAfter);
            slot.Widget.Place(Vector3.Lerp(slot.Widget.transform.position, slot.Target, k));
        }
        foreach (var id in _retired) { Destroy(_slots[id].Widget.gameObject); _slots.Remove(id); }
        _retired.Clear();
    }

#if UNITY_EDITOR
    void Reset()   // wire the prefab when the component is added
    {
        prefab = UnityEditor.AssetDatabase.LoadAssetAtPath<PortionWidget>("Assets/Relish/Prefabs/PortionWidget.prefab");
    }
#endif

    // ---- runnable check: right-click the component in Play mode. Three objects appear,
    // ---- dim after staleAfter, vanish after retireAfter — the whole lifecycle from one click.

    [ContextMenu("Test: fake frame in front of camera")]
    void TestFakeFrame()
    {
        var head = Camera.main != null ? Camera.main.transform : transform;
        var origin = head.position + head.forward * 0.6f - head.up * 0.2f;
        var frame = new Detection[3];
        for (int i = 0; i < frame.Length; i++)
            frame[i] = new Detection
            {
                Id = i,
                Position = origin + head.right * (0.12f * (i - 1)),
                Diameter = 0.04f + 0.02f * i,
                Score = 0.9f,
            };
        Report(frame);
    }

    [ContextMenu("Test: success on all")]
    void TestSuccess()
    {
        foreach (var slot in _slots.Values) slot.Widget.Success();
    }
}
