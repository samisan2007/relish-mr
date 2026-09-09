using System.Reflection;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Draws a button for every parameterless [ContextMenu] method, so the Play-mode checks
/// are one click instead of a right-click. Buttons are disabled outside Play mode because
/// the checks spawn objects and start coroutines.
/// </summary>
class ContextMenuButtonsEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var methods = target.GetType().GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        bool any = false;
        using (new EditorGUI.DisabledScope(!Application.isPlaying))
        {
            foreach (var method in methods)
            {
                var menu = method.GetCustomAttribute<ContextMenu>();
                if (menu == null || method.GetParameters().Length != 0) continue;
                if (!any) { EditorGUILayout.Space(); any = true; }
                if (GUILayout.Button(menu.menuItem))
                    foreach (var t in targets) method.Invoke(t, null);
            }
        }
        if (any && !Application.isPlaying)
            EditorGUILayout.HelpBox("Enter Play mode to use the test buttons.", MessageType.None);
    }
}

// ponytail: scoped to our two scripts. For buttons on every script, replace these with
// [CustomEditor(typeof(MonoBehaviour), true)] on the base class.
[CustomEditor(typeof(PortionWidget)), CanEditMultipleObjects] class PortionWidgetEditor : ContextMenuButtonsEditor { }
[CustomEditor(typeof(PortionWidgets)), CanEditMultipleObjects] class PortionWidgetsEditor : ContextMenuButtonsEditor { }
