using UnityEngine;
using UnityEngine.InputSystem; // <--- 1. ต้องเพิ่มบรรทัดนี้

public class BallDisturbance : MonoBehaviour
{
    [Header("Settings")]
    public float moveForce = 10f;
    public ForceMode forceMode = ForceMode.Force;

    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float h = 0f;
        float v = 0f;

        // 2. เช็ค Keyboard โดยตรง (แบบบ้านๆ ไม่ต้อง Setup Action Map)
        if (Keyboard.current != null)
        {
            if (Keyboard.current.wKey.isPressed) v += 1f;
            if (Keyboard.current.sKey.isPressed) v -= 1f;
            if (Keyboard.current.dKey.isPressed) h += 1f;
            if (Keyboard.current.aKey.isPressed) h -= 1f;
        }

        Vector3 forceDir = new Vector3(h, 0, v).normalized; // .normalized เพื่อให้เฉียงไม่แรงกว่าตรง

        if (forceDir.magnitude > 0.1f)
        {
            rb.AddForce(forceDir * moveForce, forceMode);
        }
    }
}