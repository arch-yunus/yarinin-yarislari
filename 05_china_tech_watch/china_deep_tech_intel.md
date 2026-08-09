# 🇨🇳 Doğu'nun Derin Teknoloji İstihbaratı ve Endüstri Analizi (China Tech Watch)

> **Gizlilik Derecesi: Halka Açık İstihbarat & Endüstriyel Analiz (OSINT)**  
> **Odak Alanları:** İnsansı Robotik (Unitree/AgiBot), Katı Hal Bataryaları (WeLion/CATL), RISC-V Silikon Bağımsızlığı, Alçak İrtifa Ekonomisi (eVTOL).  
> **Derleyen:** Bahattin Yunus Çetin | Çetin Deep-Tech Ar-Ge Hub

---

## 🤖 1. Bedenlenmiş Yapay Zeka & Kitlesel Robotik (Embodied AI)

Çin ekosistemi, insansı robotları (humanoids) sadece laboratuvar prototipi olarak değil, otomotiv montaj bantlarında ve ağır sanayide yer alacak fiziksel bir iş gücü olarak konumlandırmaktadır. Bu alandaki iki dev liderin teknik detayları aşağıda analiz edilmiştir.

### A. AgiBot (Zhiyuan Zhihai) ve "AimRT" & "Link-U OS" Mimarisi
Eski Huawei dahi çocuğu *Peng Zhihui* tarafından kurulan AgiBot, robot yazılım mimarisini tamamen açık kaynaklı hale getirerek bir "Android" ekosistemi yaratmaya çalışıyor.

*   **AimRT Robotik Ara Yazılımı (Middleware):** Robotik dünyasında standart olan ROS/ROS2'nin deterministik olmaması (real-time gecikmeler) ve yüksek bellek tüketimine karşı geliştirilmiştir.
    *   **Teknik Yapı:** Sıfır kopyalama (zero-copy) bellek yönetimi ve düşük gecikmeli IPC (Inter-Process Communication) sunar. Robot eklemleri arasındaki veri paketlerini mikro saniyeler düzeyinde senkronize eder.
    *   **Protokol:** ROS2 DDS mimarisi yerine doğrudan asenkron RPC ve hafif RPC kanalları kullanır.
*   **Link-U OS (Bedenlenmiş İşletim Sistemi):** 
    *   **One Body, Three Intelligences:** AgiBot donanımını üç yapay zeka katmanı yönetir:
        1.  *Etkileşim Zekası (Interaction Intelligence):* LLM/VLM tabanlı doğal dil ve görsel komut anlama.
        2.  *Görev Zekası (Task Intelligence):* Görevi alt adımlara bölen ve davranış ağaçları (behavior trees) üreten katman.
        3.  *Hareket Zekası (Motion Intelligence):* Milisaniyelik motor tork kontrolü.
*   **AgiBot X1 Açık Kaynak İnsansı Robot:** AgiBot, X1 modelinin BOM (Bill of Materials) listesini ve 3D CAD modellerini tamamen açmıştır. X1, 3D baskıyla üretilebilen uzuvlara ve 48V bus mimarisine sahip ultra ucuz bir Ar-Ge platformudur.

### B. Unitree Robotics (G1 & H1) Kontrol Algoritmaları
Dört ayaklı robotlardan insansı robotlara geçiş yapan Unitree, donanım entegrasyonu ve fiyat avantajıyla öne çıkmaktadır.

*   **Kontrol Döngüsü ve Motor Arayüzleri:**
    *   **Low-Level Kontrol:** Her bir eklem motoruna doğrudan RS485 veya CAN-FD üzerinden tork komutları (millinewton-metre cinsinden) saniyede 1000 kez (1 kHz) gönderilir.
    *   **High-Level Kontrol:** Robotun gövdesinde yer alan Jetson Orin / Intel NUC benzeri kartlar, RL (Reinforcement Learning) tabanlı hareket politikalarını çalıştırır.
*   **Isaac Sim ve MuJoCo Entegrasyonu:** Unitree, robotlarının denge ve yürüyüş modellerini Nvidia Isaac Sim üzerinde milyarlarca kez koşturarak eğitir. Simülasyonda eğitilen sinir ağları doğrudan mikrokontrolcülere aktarılmadan önce INT8/INT4 seviyesinde kuantize edilir.
*   **Eylül 2025 Siber Güvenlik Bulguları:** 
    *   Akademik araştırmacılar, Unitree G1 robotlarının Bluetooth Low Energy (BLE) eşleşme protokollerinde kritik bir açık tespit etmiştir. 
    *   Açık sayesinde, yakındaki bir saldırgan BLE üzerinden yetkisiz eşleşme sağlayarak doğrudan tork kontrol register'larına müdahale edebilmekte ve robotun dengesini bozarak düşmesine neden olabilmektedir. (Geliştirici ekibin bu açığı kapatmak için şifrelenmiş BLE kanallarına ve donanımsal el sıkışmaya (Hardware Handshake) geçiş yaptığı sızan bilgiler arasındadır).

---

## 🔋 2. Katı Hal Batarya Devrimi (Solid-State Batteries)

Fiziksel dünyadaki otonomiyi sınırlayan en büyük darboğaz enerjidir. Çin, batarya kimyasında sıvı elektrolitlerden yarı-katı ve tam-katı elektrolitlere geçişte dünyaya öncülük etmektedir.

### A. WeLion 150 kWh Yarı-Katı Batarya Paketi (Nio Ortaklığı)
Nio elektrikli araçlarında kullanılan 150 kWh kapasiteli paket, ticari olarak ölçeklendirilmiş ilk yarı-katı bataryadır.

*   **Teknik Özellikler:**
    | Parametre | Detay |
    | :--- | :--- |
    | **Kimyasal Yapı** | Hibrit katı-sıvı elektrolit, ultra yüksek nikel katot, silikon-karbon kompozit anot |
    | **Hücre Enerji Yoğunluğu** | 360 Wh/kg |
    | **Toplam Paket Ağırlığı** | 575 kg (Geleneksel 100 kWh paketlerle aynı fiziksel boyut ve ağırlıkta) |
    | **Soğutma & Isı Yönetimi** | Hücre seviyesinde termal kaçak (thermal runaway) koruması. Sıvı elektrolitin %80 oranında azaltılması yanma riskini neredeyse sıfırlamıştır. |
*   **Maliyet Dedikoduları:** Paket maliyetinin bir elektrikli otomobil üretme maliyetine eşdeğer olduğu (yaklaşık 40.000 USD) bilinmektedir. Bu sebeple Nio, bu bataryayı satın alma seçeneği sunmak yerine swap (batarya değişimi) istasyonlarında günlük kiralama modeliyle sunmaktadır.

### B. CATL Shenxing & Condensed Battery Gelişmeleri
Dünyanın en büyük batarya üreticisi CATL, 2026-2027 bandında tamamen katı hal bataryaları seri üretime geçireceğini duyurmuştur.
*   **Condensed Battery:** 500 Wh/kg hücre enerji yoğunluğuna ulaşan bu kimya, havacılık sektörü ve elektrikli uçaklar için tasarlanmıştır. Mikro boyutta polimerize edilmiş jel benzeri bir elektrolit kullanır.
*   **Termal İzleme Ağı (Smart-BMS):** Her hücre, doğrudan paket içi veri yolu üzerinden BMS (Battery Management System) tarafından milisaniyelik voltaj ve sıcaklık değişimleriyle izlenir. Olası bir dendrit oluşumu (katı hal bataryaların en büyük problemi olan lityum kristallerinin elektroliti delmesi durumu) önceden tahmin edilerek ilgili hücre grubu izole edilir.

---

## 🔌 3. RISC-V Mimarisi & Silikon Bağımsızlığı

Batı dünyasının uyguladığı x86 ve ARM ambargolarına yanıt olarak Çin, işlemci tasarımlarını tamamen açık kaynaklı **RISC-V** mimarisine kaydırmaktadır.

```
       [ Üst Seviye Uygulama (PyTorch / Link-U OS) ]
                             │
                             ▼
     [ Derleme Katmanı: Custom RV64GC-NPU Compiler ]
                             │
                             ▼
       [ İşlemci Çekirdeği: T-Head XuanTie C910 / C920 ]
                             │
                             ▼
  [ Neuromorphic Register-Level Control (bsp_register_map) ]
```

*   **T-Head XuanTie Serisi (Alibaba Group):** 
    *   *XuanTie C910 & C920:* RISC-V 64-bit yüksek performanslı işlemci çekirdekleridir. Yapay zeka ve edge bilişim için özel vektör uzantılarına (Vector Extension 1.0) sahiptir.
    *   Bu çipler, ulaştıkları yüksek saat frekanslarıyla edge kameralardan insansı robot kontrol kartlarına kadar geniş bir yelpazede kullanılmaktadır.
*   **XiangShan Projesi:** Çin Bilimler Akademisi tarafından geliştirilen, açık kaynaklı ve ARM Cortex-A76 başarımına yaklaşan ultra yüksek performanslı RISC-V işlemci mimarisidir.
*   **Özelleştirilmiş ASIC/NPU Tasarımları:** İnsansı robotlardaki eklem denetleyicileri için, sadece belirli matris çarpım fonksiyonlarını (GEMM) bare-metal düzeyde çalıştıran, donanımsal olarak gömülü RISC-V kontrolcüleri tasarlanmaktadır. Bu sayede işlemci dikeyinde güç tüketimi watt seviyelerinden miliwatt seviyelerine indirilmektedir.

---

## 🛸 4. Alçak İrtifa Ekonomisi (Low-Altitude Economy) & eVTOL

Kentsel Hava Taşımacılığı (UAM) ve otonom lojistik, Çin hükümetinin en çok desteklediği yeni ekonomik büyüme alanıdır.

### A. CAAC Düzenleyici Çerçevesi: "Önce Kargo, Sonra Yolcu"
Çin Sivil Havacılık Dairesi (CAAC), eVTOL sertifikasyon süreçlerini hızlandırmak için esnek ama aşamalı bir yol izlemektedir:
1.  **Cargo before Passengers:** İlk etapta sadece kargo drone'ları ve lojistik eVTOL'lerin uçuşuna izin verilmektedir. (AutoFlight V2000CG modeli bu kapsamda tonlarca yük taşımıştır).
2.  **Segregated before Integrated:** İlk aşamada otonom uçuşlar için özel hava koridorları (ayrılmış hava sahası) oluşturulmakta, ardından ticari uçaklarla entegre hava sahasına geçilmektedir.
3.  **Suburban before Urban:** Yoğun nüfuslu şehir merkezlerinden önce, banliyölerde ve adalar arası taşımacılıkta test uçuşları yapılmaktadır.

### B. EHang, XPeng AeroHT ve AutoFlight Teknolojileri
*   **EHang EH216-S:** Dünyada pilotsuz yolcu taşıma lisansı (Type Certificate, Production Certificate, Airworthiness Certificate) alan ilk otonom hava aracıdır. 16 pervaneli multikopter tasarımı, yedekli uçuş bilgisayarları (triple-redundant FCS) ve 5G/4G üzerinden merkezi kontrol istasyonuna bağlı otonom uçuş sistemiyle çalışır.
*   **XPeng AeroHT "Kara Uçak Gemisi":** Modüler yapıda, 6 tekerlekli karasal bir SUV içine entegre edilmiş 2 kişilik eVTOL'den oluşur. Karasal araç, hava aracını içinde taşır, şarj eder ve gerektiğinde hava aracı üstten dikey olarak kalkış yapar.
*   **UTICN Hücresel Hava Sahası Desteği:** eVTOL'lerin yer kontrol istasyonlarıyla kesintisiz haberleşebilmesi ve hava sahasındaki diğer otonom drone'ları algılayabilmesi için 5G-Advanced ağlarına entegre "Sensing and Communication" (ISAC) baz istasyonları konuşlandırılmaktadır. Bu sistem, radara ihtiyaç duymadan hava araçlarının 3D konumunu santimetre hassasiyetinde tespit eder.

---

## 🕵️ 5. Endüstriyel Sızıntılar, Dedikodular & Tedarik Zinciri Dinamikleri

*   **Galyum ve Germanyum İhracat Kısıtlamaları:** Çin hükümetinin çip yapımında kritik öneme sahip bu iki hammadde üzerindeki ihracat kontrollerini sıkılaştırması, Batı'daki yüksek frekanslı güç transistörleri (GaN/SiC) ve kızılötesi askeri sensör üretiminde darboğazlara sebep olmuştur. Sızıntılara göre, Çinli üreticiler bu ham maddeleri kendi iç pazarlarında çok daha ucuz fiyatlarla yerli RISC-V ve NPU üreticilerine sunmaktadır.
*   **Unitree vs AgiBot Yetenek Savaşı:** Huawei ve BYD'den ayrılan üst düzey kontrol mühendislerinin büyük bir gizlilikle AgiBot'a katıldığı, Unitree'nin ise batıdaki Ar-Ge merkezlerinden (özellikle Boston Dynamics ve MIT laboratuvarlarındaki araştırmacılardan) tersine mühendislik raporları topladığı dedikoduları endüstride sıkça konuşulmaktadır.
*   **Katı Hal Bataryalarında Kobalt Bağımsızlığı:** Patent başvurularına göre CATL ve WeLion, katı hal bataryalarda kobalt kullanımını tamamen sıfırlayan ve lityum yerine zengin sodyum türevleri kullanan yeni bir katı hal anot yapısı üzerinde çalışmaktadır. Bu gelişme batarya maliyetini %60 oranında düşürebilir.
