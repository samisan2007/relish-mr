using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Builds the PortionWidget prefab so the sphere, the transparent URP material and the
/// debug label are generated rather than hand-authored. Re-run it to rebuild from scratch.
/// </summary>
static class RelishWidgetBuilder
{
    const string Root = "Assets/Relish";
    const string PrefabPath = Root + "/Prefabs/PortionWidget.prefab";
    const string MaterialPath = Root + "/Materials/PortionWidget.mat";

    [MenuItem("Relish/Create Widget Prefab")]
    static void Build()
    {
        EnsureFolder(Root, "Prefabs");
        EnsureFolder(Root, "Materials");

        var material = BuildMaterial();

        var root = new GameObject("PortionWidget");
        var widget = root.AddComponent<PortionWidget>();

        var ball = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        ball.name = "Ball";
        ball.transform.SetParent(root.transform, false);
        var ballRenderer = ball.GetComponent<MeshRenderer>();
        ballRenderer.sharedMaterial = material;
        ballRenderer.shadowCastingMode = ShadowCastingMode.Off;

        // TMP swaps in a RectTransform, so add it before parenting.
        var labelGO = new GameObject("Label");
        var label = labelGO.AddComponent<TextMeshPro>();
        label.transform.SetParent(root.transform, false);
        label.text = "";
        label.fontSize = 0.3f;
        label.alignment = TextAlignmentOptions.Center;
        label.rectTransform.sizeDelta = new Vector2(0.6f, 0.12f);
        labelGO.SetActive(false);

        var so = new SerializedObject(widget);
        so.FindProperty("ball").objectReferenceValue = ball.transform;
        so.FindProperty("ballRenderer").objectReferenceValue = ballRenderer;
        so.FindProperty("label").objectReferenceValue = label;
        so.ApplyModifiedPropertiesWithoutUndo();

        PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        Object.DestroyImmediate(root);

        if (TMP_Settings.instance == null || TMP_Settings.defaultFontAsset == null)
            Debug.LogWarning("TMP Essentials are not imported, so the debug label will not render. " +
                             "Window > TextMeshPro > Import TMP Essential Resources.");

        var asset = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        Selection.activeObject = asset;
        EditorGUIUtility.PingObject(asset);
        Debug.Log($"Built {PrefabPath}", asset);
    }

    /// <summary>URP Lit, set to transparent so the widget does not hide the food behind it.</summary>
    static Material BuildMaterial()
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            Debug.LogError("URP Lit shader not found — is the project still on URP?");
            return null;
        }

        var material = new Material(shader);
        material.SetFloat("_Surface", 1f);            // 0 opaque, 1 transparent
        material.SetFloat("_Blend", 0f);              // alpha blend
        BaseShaderGUI.SetupMaterialBlendMode(material); // URP applies blend state, keywords and queue from those two
        material.SetColor("_BaseColor", new Color(0.25f, 0.75f, 1f, 0.35f));  // runtime overrides this

        AssetDatabase.DeleteAsset(MaterialPath);
        AssetDatabase.CreateAsset(material, MaterialPath);
        return material;
    }

    static void EnsureFolder(string parent, string child)
    {
        if (!AssetDatabase.IsValidFolder(parent)) AssetDatabase.CreateFolder("Assets", "Relish");
        if (!AssetDatabase.IsValidFolder($"{parent}/{child}")) AssetDatabase.CreateFolder(parent, child);
    }
}
