using UnityEngine;

public class InverseKine : MonoBehaviour
{
    [Header("References")]
    public JointControl jointControl;

    [Header("--- Calibration Mode ---")]
    [Tooltip("ติ๊กถูกเพื่อบังคับให้หุ่นอยู่นิ่งๆ ตรงกลางสำหรับการจูน")]
    public bool calibrationMode = true; 

    [Header("Platform Parameters")]
    public float platformRadius = 0.1f;  
    public float baseRadius = 0.2f;      
    public float linkLength_ca = 0.3f;    
    public float linkLength_bc = 0.4f;    

    [Header("Debug")]
    [SerializeField] private bool verboseGeometryLogs = false;

    // ซ่อนให้ Inspector ไม่รก (เพราะเราจะคำนวณอัตโนมัติ)
    private Vector3[] r_a = new Vector3[3];
    private Vector3[] r_b = new Vector3[3];
    private Vector3[] y_prime = new Vector3[3];

    [Header("Target Pose (Read Only in Calib Mode)")]
    public Vector3 targetPosition = new Vector3(0, 0.35f, 0); // เริ่มที่ 0.35 ตามที่คุณตั้ง
    public Vector3 targetRotation = Vector3.zero;            

    [Header("Motor Output")]
    public float[] motorAngle_qbi = new float[3];
    public float[] motorAngle_qci = new float[3];

    // Event Functions (ปล่อยไว้เหมือนเดิม แต่จะไม่ทำงานถ้า Calibration Mode เปิดอยู่)
    public void SetTargetRotationX(float xGain) { if(!calibrationMode) targetRotation.x = Mathf.Clamp(-xGain, -20f, 20f); }
    public void SetTargetRotationZ(float zGain) { if(!calibrationMode) targetRotation.z = Mathf.Clamp(-zGain, -20f, 20f); }

    void Update()
    {
        // 1. บังคับคำนวณ Geometry ใหม่ทุกเฟรม (เผื่อคุณปรับ Radius เล่น)
        SetupGeometry();

        // 2. ถ้าอยู่ในโหมดจูน ให้บังคับ Rotation เป็น 0
        if (calibrationMode)
        {
            targetRotation = Vector3.zero;
        }

        // 3. คำนวณ IK
        CalculateAllIK();
            
        // 4. ส่งค่าไปขับหุ่น
        if (jointControl != null)
        {
            jointControl.SetLegAngles(motorAngle_qbi);
        }
    }

    void SetupGeometry()
    {
        float[] angles = { 0f, 120f, 240f };
        for (int i = 0; i < 3; i++)
        {
            float rad = angles[i] * Mathf.Deg2Rad;
            // คำนวณเวกเตอร์ให้สมมาตร 100%
            r_a[i] = new Vector3(Mathf.Cos(rad) * platformRadius, 0, Mathf.Sin(rad) * platformRadius);
            r_b[i] = new Vector3(Mathf.Cos(rad) * baseRadius, 0, Mathf.Sin(rad) * baseRadius);
            y_prime[i] = new Vector3(-Mathf.Sin(rad), 0, Mathf.Cos(rad));
            if (verboseGeometryLogs)
            {
                Debug.Log($"r_a[{i}]: {r_a[i]}, r_b[{i}]: {r_b[i]}, y_prime[{i}]: {y_prime[i]}", this);
            }
        }
    }

    public void CalculateLegIK(int i)
    {
        // คำนวณ IK ตามสูตรเดิม
        Quaternion R_op_Quat = Quaternion.Euler(targetRotation.x, targetRotation.y, targetRotation.z);
        Vector3 rotated_r_ai = R_op_Quat * r_a[i];
        Vector3 L_bai_Vector = targetPosition + rotated_r_ai - r_b[i];

        float L_bai = L_bai_Vector.magnitude;
        float l_bci = linkLength_bc;
        float l_cai = linkLength_ca;

        float cos_alpha_i_numerator = (l_bci * l_bci) + (L_bai * L_bai) - (l_cai * l_cai);
        float cos_alpha_i_denominator = 2f * l_bci * L_bai;
        float cos_alpha_i = Mathf.Clamp(cos_alpha_i_numerator / cos_alpha_i_denominator, -1f, 1f);
        float alpha_i = Mathf.Acos(cos_alpha_i);

        float cos_gamma_i_numerator = (l_cai * l_cai) + (l_bci * l_bci) - (L_bai * L_bai);
        float cos_gamma_i_denominator = 2f * l_cai * l_bci;
        float cos_gamma_i = Mathf.Clamp(cos_gamma_i_numerator / cos_gamma_i_denominator, -1f, 1f);
        float gamma_i = Mathf.Acos(cos_gamma_i);
        
        float dot_product = Vector3.Dot(L_bai_Vector, y_prime[i]);
        float cos_beta_i_denominator = L_bai * y_prime[i].magnitude; 

        float cos_beta_i = 0f;
        if (cos_beta_i_denominator != 0)
            cos_beta_i = Mathf.Clamp(dot_product / cos_beta_i_denominator, -1f, 1f);
        
        float beta_i = Mathf.Acos(cos_beta_i); 
                                               
        float q_bi_Radian = alpha_i + beta_i - (Mathf.PI / 2f);
        motorAngle_qbi[i] = q_bi_Radian * Mathf.Rad2Deg; 
        
        float q_ci_Radian = gamma_i - Mathf.PI;
        motorAngle_qci[i] = q_ci_Radian * Mathf.Rad2Deg; 
    }

    public void CalculateAllIK()
    {
        for (int i = 0; i < 3; i++) CalculateLegIK(i);
    }
}
