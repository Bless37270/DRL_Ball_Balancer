using UnityEngine;

public class BallSpawner : MonoBehaviour
{
    [Header("Settings")]
    public GameObject ballPrefab; // ลาก Prefab ลูกบอลมาใส่ที่นี่
    public float spawnRadius = 10f; // รัศมีวงกลม (10 หน่วย)
    public float minSpawnHeight = 16f; // ความสูงต่ำสุด (16 หน่วย)
    public float maxSpawnHeight = 22f; // ความสูงสูงสุด (22 หน่วย)
    public float heightThreshold = 3f; // ความสูงที่ถือว่าตก (3 หน่วย)
    public float timeLimit = 20f; // เวลาจำกัด (20 วินาที)

    [Header("Stats (Read Only)")]
    [SerializeField] private int totalSpawnCount = 0; // จำนวนที่สุ่มไปทั้งหมด
    [SerializeField] private int droppedCount = 0; // จำนวนครั้งที่ตกต่ำกว่า 3 หน่วย

    private GameObject currentBall; // ตัวแปรเก็บลูกบอลปัจจุบัน
    private float timer; // ตัวจับเวลา

    void Start()
    {
        SpawnNewBall();
    }

    void Update()
    {
        // ถ้าไม่มีลูกบอล (อาจจะเพิ่งถูกลบไป) ให้ข้ามการทำงาน
        if (currentBall == null) return;

        // อัปเดตเวลา
        timer += Time.deltaTime;

        // เงื่อนไขที่ 1: เช็คว่าความสูงต่ำกว่ากำหนดหรือไม่ (ตกพื้น)
        if (currentBall.transform.position.y < heightThreshold)
        {
            droppedCount++; // นับสถิติว่าตกสำเร็จ
            Debug.Log($"<color=green>Ball Dropped!</color> (Height < {heightThreshold}). Spawns: {totalSpawnCount}, Dropped: {droppedCount}");
            
            RemoveAndRespawn();
        }
        // เงื่อนไขที่ 2: เช็คว่าหมดเวลา 20 วินาทีหรือไม่
        else if (timer >= timeLimit)
        {
            Debug.Log($"<color=red>Time Up!</color> (20s passed). Ball reset. Spawns: {totalSpawnCount}, Dropped: {droppedCount}");
            
            RemoveAndRespawn();
        }
    }

    void SpawnNewBall()
    {
        // 1. คำนวณตำแหน่ง X, Z ในวงกลมรัศมี 10
        Vector2 randomCircle = Random.insideUnitCircle * spawnRadius;
        
        // 2. คำนวณความสูง Y ระหว่าง 16-22
        float randomHeight = Random.Range(minSpawnHeight, maxSpawnHeight);

        // 3. รวมเป็นตำแหน่ง Vector3
        Vector3 spawnPosition = new Vector3(randomCircle.x, randomHeight, randomCircle.y);

        // 4. สร้างลูกบอล
        currentBall = Instantiate(ballPrefab, spawnPosition, Quaternion.identity);
        
        // 5. รีเซ็ตค่าและนับจำนวน
        timer = 0f;
        totalSpawnCount++;

        Debug.Log($"Spawned Ball #{totalSpawnCount} at: {spawnPosition}");
    }

    void RemoveAndRespawn()
    {
        if (currentBall != null)
        {
            Destroy(currentBall); // ลบลูกบอลเก่า
        }
        SpawnNewBall(); // สร้างลูกใหม่ทันที
    }

    // ฟังก์ชันนี้ช่วยวาดเส้นในหน้า Scene ให้เห็นขอบเขตการเกิด (Debug Visual)
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        // วาดวงกลมที่ความสูงต่ำสุด
        DrawCircle(transform.position + Vector3.up * minSpawnHeight, spawnRadius);
        // วาดวงกลมที่ความสูงสูงสุด
        DrawCircle(transform.position + Vector3.up * maxSpawnHeight, spawnRadius);
        
        Gizmos.color = Color.red;
        // วาดเส้นความสูงที่ถือว่าตก
        Gizmos.DrawWireCube(new Vector3(0, heightThreshold, 0), new Vector3(spawnRadius*2, 0.1f, spawnRadius*2));
    }

    void DrawCircle(Vector3 center, float radius)
    {
        Gizmos.DrawWireSphere(center, radius);
    }
}