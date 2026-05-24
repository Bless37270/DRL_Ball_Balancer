using UnityEngine;

public class PIDController : MonoBehaviour
{
    [Header("PID Gains (Inputs)")]
    public float Kp = 1f;
    public float Ki = 0.1f;
    public float Kd = 0.01f;

    // --- ส่องดูค่าที่คำนวณได้ ---
    [Header("Debug Values (Outputs)")]
    [SerializeField] private float p_term; // Proportional Term
    [SerializeField] private float i_term; // Integral Term
    [SerializeField] private float d_term; // Derivative Term
    [SerializeField] private float currentErrorForDebug; // ไว้ดู Error ล่าสุด

    // --- ตัวแปรภายใน ---
    private float integralSum = 0f;
    private float previousError = 0f;

    public float CalculateOutput(float currentError)
    {
        currentErrorForDebug = currentError; // อัปเดตค่าโชว์

        // 1. P
        p_term = Kp * currentError;

        // 2. I
        integralSum += currentError * Time.fixedDeltaTime;
        i_term = Ki * integralSum;

        // 3. D
        float derivative = (currentError - previousError) / Time.fixedDeltaTime;
        d_term = Kd * derivative;
        
        previousError = currentError;

        // 5. รวมค่า
        float outputGain = p_term + i_term + d_term;
        return outputGain;
    }

    /// <summary>
    /// ใช้สำหรับ Reset ค่าสะสมของ Controller (เช่น เมื่อเริ่มใหม่)
    /// </summary>
    public void ResetController()
    {
        integralSum = 0f;
        previousError = 0f;
        p_term = 0f;
        i_term = 0f;
        d_term = 0f;
    }

    // เมื่อ Script ถูกปิดการใช้งาน (Disable) ให้ Reset ค่า
    private void OnDisable()
    {
        ResetController();
    }
}