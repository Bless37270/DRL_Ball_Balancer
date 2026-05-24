using UnityEngine;

public class JointControl : MonoBehaviour
{
    [Header("--- Reference ---")]
    public Transform leg1Transform;
    public Transform leg2Transform;
    public Transform leg3Transform;

    [Header("--- Mapping Settings (IK -> Unity Angle) ---")]
    // ค่าที่คำนวณมาล่าสุดของคุณ (Input 69-159)
    public float inputLow = 69f; 
    public float inputHigh = 159f;

    public float outputLow = -45f;
    public float outputHigh = 45f;

    [Header("--- Servo Dynamics (New!) ---")]
    [Tooltip("ความหน่วงของมอเตอร์ (ค่ายิ่งมาก ยิ่งนุ่มแต่ช้า) แนะนำ 0.05 - 0.1")]
    public float smoothTime = 0.05f; 
    [Tooltip("ความเร็วสูงสุดที่มอเตอร์หมุนได้ (องศา/วินาที)")]
    public float maxSpeed = 600f; 

    [Header("--- Manual Testing ---")]
    public bool useManualControl = false;
    [Range(0f, 200f)] public float knobInput1 = 114f;
    [Range(0f, 200f)] public float knobInput2 = 114f;
    [Range(0f, 200f)] public float knobInput3 = 114f;

    [Header("--- Monitor ---")]
    [SerializeField] private float mappedAngle1;
    [SerializeField] private float mappedAngle2;
    [SerializeField] private float mappedAngle3;

    // ตัวแปรจำค่าแกนเดิม
    private Vector3 initLeg1Euler;
    private Vector3 initLeg2Euler;
    private Vector3 initLeg3Euler;

    // ตัวแปรรับค่าจาก IK
    private float ikInput1, ikInput2, ikInput3;

    // ตัวแปรสำหรับคำนวณความนุ่มนวล (SmoothDamp Velocity)
    private float velocity1, velocity2, velocity3;
    private float currentAngle1, currentAngle2, currentAngle3;

    void Start()
    {
        // จำค่ามุมเริ่มต้น
        if (leg1Transform) { initLeg1Euler = leg1Transform.localEulerAngles; currentAngle1 = 0; }
        if (leg2Transform) { initLeg2Euler = leg2Transform.localEulerAngles; currentAngle2 = 0; }
        if (leg3Transform) { initLeg3Euler = leg3Transform.localEulerAngles; currentAngle3 = 0; }
    }

    public void SetLegAngles(float[] angles)
    {
        if (angles.Length >= 3)
        {
            ikInput1 = angles[0];
            ikInput2 = angles[1];
            ikInput3 = angles[2];
        }
    }

    void FixedUpdate()
    {
        float targetRaw1, targetRaw2, targetRaw3;

        // 1. เลือก Input
        if (useManualControl)
        {
            targetRaw1 = knobInput1;
            targetRaw2 = knobInput2;
            targetRaw3 = knobInput3;
        }
        else
        {
            targetRaw1 = ikInput1;
            targetRaw2 = ikInput2;
            targetRaw3 = ikInput3;

            // Sync กลับไปที่ Knob เพื่อดูค่า
            knobInput1 = ikInput1;
            knobInput2 = ikInput2;
            knobInput3 = ikInput3;
        }

        // 2. คำนวณเป้าหมาย (Target Angle)
        float targetAngle1 = MapValue(targetRaw1, inputLow, inputHigh, outputLow, outputHigh);
        float targetAngle2 = MapValue(targetRaw2, inputLow, inputHigh, outputLow, outputHigh);
        float targetAngle3 = MapValue(targetRaw3, inputLow, inputHigh, outputLow, outputHigh);

        // 3. ใส่ความสมูท (Servo Simulation) <--- หัวใจสำคัญของการแก้สั่น!
        // แทนที่จะไปทันที เราจะค่อยๆ ขยับเข้าหาเป้าหมาย
        currentAngle1 = Mathf.SmoothDampAngle(currentAngle1, targetAngle1, ref velocity1, smoothTime, maxSpeed);
        currentAngle2 = Mathf.SmoothDampAngle(currentAngle2, targetAngle2, ref velocity2, smoothTime, maxSpeed);
        currentAngle3 = Mathf.SmoothDampAngle(currentAngle3, targetAngle3, ref velocity3, smoothTime, maxSpeed);

        // อัปเดตค่า Monitor
        mappedAngle1 = currentAngle1;
        mappedAngle2 = currentAngle2;
        mappedAngle3 = currentAngle3;

        // 4. สั่งหมุนขาจริง
        RotateLegLocked(leg1Transform, currentAngle1, initLeg1Euler);
        RotateLegLocked(leg2Transform, currentAngle2, initLeg2Euler);
        RotateLegLocked(leg3Transform, currentAngle3, initLeg3Euler);
    }

    float MapValue(float value, float from1, float to1, float from2, float to2)
    {
        float t = Mathf.InverseLerp(from1, to1, value);
        float mapped = Mathf.Lerp(from2, to2, t);
        return Mathf.Clamp(mapped, Mathf.Min(from2, to2), Mathf.Max(from2, to2));
    }

    void RotateLegLocked(Transform leg, float newZ, Vector3 initialEuler)
    {
        if (leg != null)
        {
            leg.localRotation = Quaternion.Euler(initialEuler.x, initialEuler.y, newZ);
        }
    }
}