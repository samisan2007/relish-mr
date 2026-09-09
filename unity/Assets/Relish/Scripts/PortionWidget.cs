using System.Collections;
using TMPro;
using UnityEngine;

/// <summary>
/// Placeholder portion widget: a translucent sphere sized in real-world metres.
/// A sphere for now — swap the Ball child's mesh later without touching this API.
/// Build the prefab with Relish > Create Widget Prefab.
/// </summary>
[DisallowMultipleComponent]
public class PortionWidget : MonoBehaviour
{
    [Header("Parts (wired by the prefab builder)")]
    [SerializeField] Transform ball;
    [SerializeField] Renderer ballRenderer;
    [SerializeField] TextMeshPro label;

    [Header("Size, in metres")]
    [SerializeField] float diameter = 0.08f;
    [SerializeField] float minimizedDiameter = 0.02f;
    [SerializeField] float tweenSeconds = 0.15f;

    [Header("Look")]
    [SerializeField] Color idleColor = new Color(0.25f, 0.75f, 1f, 0.35f);
    [SerializeField] Color successColor = new Color(0.3f, 1f, 0.45f, 0.7f);
    [SerializeField] Color staleColor = new Color(0.6f, 0.6f, 0.6f, 0.15f);
    [SerializeField] float successSeconds = 0.6f;
    [SerializeField] float successPulse = 0.15f;

    [Header("Debug")]
    [SerializeField] bool drawGizmos = true;

    static readonly int BaseColor = Shader.PropertyToID("_BaseColor");

    public float Diameter => diameter;
    public bool IsMinimized { get; private set; }

    /// <summary>What the ball should currently be scaled to.</summary>
    float Target => IsMinimized ? minimizedDiameter : diameter;

    /// <summary>Colour when nothing is animating.</summary>
    Color RestColor => _stale ? staleColor : idleColor;

    MaterialPropertyBlock _block;
    Coroutine _tween, _flash;
    bool _stale;
    Transform _head;

    void Awake()
    {
        SetColor(RestColor);
        ApplyScale(Target);
    }

    void OnDisable()
    {
        // Coroutines die with the object; drop the handles or Retarget stays blocked.
        _tween = _flash = null;
        SetColor(RestColor);
        ApplyScale(Target);
    }

    void LateUpdate()
    {
        // Keep the debug label facing the viewer.
        if (label == null || !label.gameObject.activeSelf) return;
        if (_head == null)
        {
            var cam = Camera.main;
            if (cam == null) return;
            _head = cam.transform;
        }
        label.transform.rotation = Quaternion.LookRotation(label.transform.position - _head.position);
    }

    // ---- the API the perception bridge will call ----

    public void Show() => gameObject.SetActive(true);
    public void Hide() => gameObject.SetActive(false);

    public void Place(Vector3 worldPosition) => transform.position = worldPosition;
    public void Place(Pose pose) => transform.SetPositionAndRotation(pose.position, pose.rotation);

    /// <summary>Size the widget to a real-world diameter in metres.</summary>
    public void SetDiameter(float metres)
    {
        diameter = Mathf.Max(0.001f, metres);
        Retarget();
    }

    public void Minimize() { IsMinimized = true; Retarget(); }
    public void Restore() { IsMinimized = false; Retarget(); }
    public void SetMinimized(bool minimized) { IsMinimized = minimized; Retarget(); }

    public void SetLabel(string text)
    {
        if (label == null) return;
        label.text = text;
        label.gameObject.SetActive(!string.IsNullOrEmpty(text));
    }

    /// <summary>Dim while perception has not seen this object recently.</summary>
    public void SetStale(bool stale)
    {
        if (_stale == stale) return;
        _stale = stale;
        if (_flash == null) SetColor(RestColor);
    }

    /// <summary>Green pulse — confirmation that a portion was accepted.</summary>
    public void Success()
    {
        if (!isActiveAndEnabled) return;
        StopTween();
        if (_flash != null) StopCoroutine(_flash);
        _flash = StartCoroutine(FlashRoutine());
    }

    // ---- internals ----

    void Retarget()
    {
        if (!isActiveAndEnabled) { ApplyScale(Target); return; }
        if (_flash != null) return;          // the flash lands on Target itself
        StopTween();
        _tween = StartCoroutine(ScaleRoutine(Target));
    }

    void StopTween()
    {
        if (_tween != null) StopCoroutine(_tween);
        _tween = null;
    }

    IEnumerator ScaleRoutine(float to)
    {
        float from = ball != null ? ball.localScale.x : to;
        for (float t = 0f; t < tweenSeconds; t += Time.deltaTime)
        {
            ApplyScale(Mathf.Lerp(from, to, t / tweenSeconds));
            yield return null;
        }
        ApplyScale(to);
        _tween = null;
    }

    IEnumerator FlashRoutine()
    {
        for (float t = 0f; t < successSeconds; t += Time.deltaTime)
        {
            float p = t / successSeconds;
            SetColor(Color.Lerp(successColor, RestColor, p));
            ApplyScale(Target * (1f + successPulse * Mathf.Sin(p * Mathf.PI)));
            yield return null;
        }
        SetColor(RestColor);
        ApplyScale(Target);
        _flash = null;
    }

    void ApplyScale(float d)
    {
        // The built-in sphere is 1m across at scale 1, so scale == diameter.
        if (ball != null) ball.localScale = Vector3.one * d;
        if (label != null) label.transform.localPosition = new Vector3(0f, d * 0.5f + 0.03f, 0f);
    }

    void SetColor(Color color)
    {
        if (ballRenderer == null) return;
        _block ??= new MaterialPropertyBlock();
        ballRenderer.GetPropertyBlock(_block);
        _block.SetColor(BaseColor, color);
        ballRenderer.SetPropertyBlock(_block);
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        Gizmos.color = new Color(idleColor.r, idleColor.g, idleColor.b, 0.9f);
        Gizmos.DrawWireSphere(transform.position, Target * 0.5f);
        Gizmos.color = Color.white;
        Gizmos.DrawRay(transform.position, transform.up * (Target * 0.5f + 0.03f));

        if (!Application.isPlaying) return;
        Gizmos.color = Color.yellow;
        foreach (var hand in Hands.Tracked)
            Gizmos.DrawLine(transform.position, hand.PointerPose.position);
    }

    // ---- runnable checks: right-click the component in Play mode ----

    [ContextMenu("Test success")]
    void TestSuccess() => Success();

    [ContextMenu("Test minimize toggle")]
    void TestMinimize() => SetMinimized(!IsMinimized);

    [ContextMenu("Test place at right pinch")]
    void TestPlace()
    {
        if (Hands.TryGetPinchPoint(OVRPlugin.Hand.HandRight, out var p)) Place(p);
        else Debug.Log("PortionWidget: right hand is not pinching.", this);
    }
}
