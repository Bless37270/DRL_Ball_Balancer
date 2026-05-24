using UnityEngine;
using UnityEngine.Events;

[System.Serializable]
public class GainCalculatedEvent : UnityEvent<float> {}

public class Controller : MonoBehaviour
{
    [Header("PID Controllers")]
    public PIDController pidControllerX;
    public PIDController pidControllerZ;

    [Header("Ref: The Plant")]
    public RealBallDetector ballSensor;
    public Transform plateTransform; // <-- 1. ลากตัว Plate (แผ่นบน) มาใส่ตรงนี้
    public Transform ballTransform;

    [Header("Settings")]
    public Vector2 targetPosition = Vector2.zero;
    public bool useTransformMeasurement = true;
    public bool fallbackToCameraMeasurement = true;

    [Header("--- Debug Graph (Pro Tip) ---")]
    // ใช้ AnimationCurve เพื่อพล็อตกราฟใน Inspector ได้เลย!
    public bool showDebugGraph = true;
    public AnimationCurve graphPidX = new AnimationCurve();
    public AnimationCurve graphPlateX = new AnimationCurve();

    [Header("--- Realtime Monitor ---")]
    [SerializeField] private float pidOutputX; // ค่าที่ PID สั่ง
    [SerializeField] private float pidOutputZ;
    [SerializeField] private float actualPlateX; // ค่าองศาจริงของเพลท
    [SerializeField] private float actualPlateZ;
    [SerializeField] private float diffX;      // ความต่าง (Lag)
    [SerializeField] private float sensedX;
    [SerializeField] private float sensedZ;
    [SerializeField] private bool usingTransformMeasurement;

    [Header("Events (Output)")]
    public GainCalculatedEvent OnGainCalculatedX;
    public GainCalculatedEvent OnGainCalculatedZ;

    private float timer;

    void Start()
    {
        ResolveReferences();
        if (plateTransform == null) Debug.LogError("อย่าลืมลาก Plate Transform มาใส่ครับ!");
    }

    void FixedUpdate()
    {
        ResolveReferences();
        if (plateTransform == null || pidControllerX == null || pidControllerZ == null) return;
        if (!TryGetBallMeasurement(out Vector2 measuredPosition)) return;

        // --- 1. คำนวณ PID ตามปกติ ---
        sensedX = measuredPosition.x;
        sensedZ = measuredPosition.y;

        float errorX = targetPosition.x - sensedX;
        float errorZ = targetPosition.y - sensedZ;

        float gainX = pidControllerX.CalculateOutput(errorX);
        float gainZ = pidControllerZ.CalculateOutput(errorZ);

        // --- 2. อ่านค่าองศาจริงของ Plate ---
        // ต้องแปลงมุมเพราะ Unity ส่งค่า 0..360 แต่เราอยากได้ -180..180
        float currentPlateX = ValidateAngle(plateTransform.localEulerAngles.x);
        float currentPlateZ = ValidateAngle(plateTransform.localEulerAngles.z);

        // --- 3. อัปเดต Monitor ---
        pidOutputX = gainX;
        pidOutputZ = gainZ;
        actualPlateX = currentPlateX;
        actualPlateZ = currentPlateZ;
        diffX = gainX - currentPlateX; // ดูว่ามันห่างกันเยอะไหม

        // --- 4. Plot Graph ลง Inspector (เทคนิคลับ) ---
        if (showDebugGraph)
        {
            timer += Time.fixedDeltaTime;
            // พล็อตค่า PID (สีแดงในจินตนาการ)
            graphPidX.AddKey(timer, gainX);
            // พล็อตค่าจริง (สีเขียวในจินตนาการ)
            graphPlateX.AddKey(timer, currentPlateX);

            // เคลียร์กราฟถ้ายาวเกินไป (กันเมมเต็ม)
            if (graphPidX.length > 500)
            {
                graphPidX = new AnimationCurve();
                graphPlateX = new AnimationCurve();
                timer = 0;
            }
        }

        // --- 5. ส่งค่าออกไป ---
        OnGainCalculatedX.Invoke(gainX);
        OnGainCalculatedZ.Invoke(gainZ);
    }

    // ฟังก์ชันแปลงมุม 0..360 ให้เป็น -180..180
    // เช่น 359 -> -1,  10 -> 10
    float ValidateAngle(float angle)
    {
        if (angle > 180) angle -= 360;
        return angle;
    }

    private bool TryGetBallMeasurement(out Vector2 measuredPosition)
    {
        if (useTransformMeasurement && ballTransform != null && plateTransform != null)
        {
            Vector3 localBallPosition = plateTransform.InverseTransformPoint(ballTransform.position);
            measuredPosition = new Vector2(localBallPosition.x, localBallPosition.z);
            usingTransformMeasurement = true;
            return true;
        }

        if (fallbackToCameraMeasurement && ballSensor != null && ballSensor.HasBall)
        {
            measuredPosition = new Vector2(ballSensor.output_X, ballSensor.output_Y);
            usingTransformMeasurement = false;
            return true;
        }

        measuredPosition = Vector2.zero;
        return false;
    }

    private void ResolveReferences()
    {
        if (ballTransform != null || plateTransform == null)
        {
            return;
        }

        Rigidbody[] bodies = FindObjectsByType<Rigidbody>(FindObjectsSortMode.None);
        foreach (Rigidbody body in bodies)
        {
            if (body.transform == plateTransform)
            {
                continue;
            }

            ballTransform = body.transform;
            return;
        }
    }
}
