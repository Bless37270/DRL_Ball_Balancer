using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class RobotActuator : MonoBehaviour
{
    private Rigidbody selfRb;
    
    // ตัวแปรสำหรับ "เก็บ" มุมเป้าหมายที่ได้รับจาก Events
    private float targetAngleX = 0f;
    private float targetAngleZ = 0f;
    
    // เก็บมุม Y เริ่มต้นไว้ (กัน Plane หมุนไปด้านข้าง)
    private float startAngleY = 0f;

    void Start()
    {
        selfRb = GetComponent<Rigidbody>();
        startAngleY = selfRb.rotation.eulerAngles.y;
    }

    // ฟังก์ชันนี้รับค่า "มุม" จาก Error แกน Z
    public void SetTargetAngleX(float angle)
    {
        targetAngleX = angle;
    }

    // ฟังก์ชันนี้รับค่า "มุม" จาก Error แกน X
    public void SetTargetAngleZ(float angle)
    {
        targetAngleZ = angle;
    }

    // เราจะทำการหมุนใน FixedUpdate
    void FixedUpdate()
    {
        // 1. สร้าง Quaternion (ทิศทางการหมุน) จากมุมเป้าหมาย
        Quaternion targetRotation = Quaternion.Euler(targetAngleX, startAngleY, targetAngleZ);

        // 2. สั่งหมุน Rigidbody ไปยังทิศทางนั้น
        // นี่คือวิธี "หมุนโดยตรง" ที่ถูกต้องสำหรับ Rigidbody
        selfRb.MoveRotation(targetRotation);
    }
}