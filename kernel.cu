	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include <stdio.h>
	// cuda için gerekli kütüphaneler

	#include <random> // Rastgele sayı üretimi için profesyonel kütüphane
	#include <string> //dosya isimleri için
	#include <fstream> //Görseldeki pixelleri byte-byte çekmek için gerekli kütüphane
	#include <iostream>//temel fonksiyonlar için giriş çıkış kütüphanesi
	#include <thread> 
	#include <chrono> // Zaman birimleri için
	#include <vector> // Dinamik bellek yönetimi için gerekli kütüphane
	#include <cmath> // Matamatik işlemleri için gerekli kütüphane
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)


	using namespace std;
	void HeaderOku(vector<uint8_t>& headerBuffer, int& headersizeBuffer, int& GenişlikBuffer, int& YükseklikBuffer, int& paddingBuffer, ifstream& görsel) {
		görsel.seekg(10, ios::beg);
		görsel.read((char*)&headersizeBuffer, 4);
		headerBuffer.resize(headersizeBuffer);
		görsel.seekg(0, ios::beg);
		görsel.read((char*)headerBuffer.data(), headersizeBuffer);
		GenişlikBuffer = *(int*)&headerBuffer[18];
		YükseklikBuffer = *(int*)&headerBuffer[22];
		paddingBuffer = 4 - ((GenişlikBuffer * 3) % 4);
		if (paddingBuffer == 4) {
			paddingBuffer = 0;
		}
	}


	__device__ inline float Sigmoid(float x) {
		return 1.0f / (1.0f + expf(-x));
	}
	//sigmoid fonksiyonunu global içerisinde kullanabilmek için tanımlıyoruz
	__global__ void IlkIleriYayilimKernel(float* d_GirişNöronları, float* d_AraNöronlar, float* d_Biaslar, float* İlkKatmanAğırlıkları, int ToplamGirişNöronuSayısı, int ToplamAraNöronSayısı) {
		int AraNöronNo = blockIdx.x * blockDim.x + threadIdx.x;
		if (AraNöronNo >= ToplamAraNöronSayısı) { return; }
		float temp = 0.0f;
		for (int i = 0; i < ToplamGirişNöronuSayısı; i++) {
			temp += d_GirişNöronları[i] * İlkKatmanAğırlıkları[AraNöronNo * ToplamGirişNöronuSayısı + i];
		}
		// Bias Ekle
		temp += d_Biaslar[AraNöronNo];
		// Eğer temp negatifse 0.0f yazar, pozitifse kendini yazar(fmaxf fonksiyonunun özlelliği)
		d_AraNöronlar[AraNöronNo] = fmaxf(0.1f * temp, temp); // ReLU
	}
	void İlkİleriYayılım(float* d_GirişNöronları, float* d_AraNöronlar, float* d_Biaslar, float* İlkKatmanAğırlıkları, int ToplamGirişNöronuSayısı, int ToplamAraNöronSayısı) {
		int blocksize = 256;
		int BlokSayısı = (ToplamAraNöronSayısı + blocksize - 1) / blocksize;//Gereken blok sayısını hesapla
		cudaDeviceSynchronize();
		IlkIleriYayilimKernel << <BlokSayısı, blocksize >> > (d_GirişNöronları, d_AraNöronlar, d_Biaslar, İlkKatmanAğırlıkları, ToplamGirişNöronuSayısı, ToplamAraNöronSayısı);
		cudaDeviceSynchronize();
	}
	__global__ void IkinciIleriYayilimKernel(float* d_AraNöronlar, float* d_ÇıkışNöronları, float* d_Biaslar, float* İkinciKatmanAğırlıkları, int ToplamAraNöronSayısı, int ToplamÇıkışNöronuSayısı) {
		int ÇıkışNöronNo = blockIdx.x * blockDim.x + threadIdx.x;
		if (ÇıkışNöronNo >= ToplamÇıkışNöronuSayısı) { return; }
		float temp = 0.0f;
		for (int i = 0; i < ToplamAraNöronSayısı; i++) {
			temp += d_AraNöronlar[i] * İkinciKatmanAğırlıkları[ÇıkışNöronNo * ToplamAraNöronSayısı + i];
		}
		// Bias Ekle
		temp += d_Biaslar[ÇıkışNöronNo];
		d_ÇıkışNöronları[ÇıkışNöronNo] = Sigmoid(temp);//sigmoid uyguladık 
	}
	void İkinciİleriYayılım(float* d_AraNöronlar, float* d_ÇıkışNöronları, float* d_Biaslar, float* İkinciKatmanAğırlıkları, int ToplamAraNöronSayısı, int ToplamÇıkışNöronuSayısı) {
		int blocksize = 256;
		int BlokSayısı = (ToplamÇıkışNöronuSayısı + blocksize - 1) / blocksize;//Gereken blok sayısını hesapla
		cudaDeviceSynchronize();
		IkinciIleriYayilimKernel << <BlokSayısı, blocksize >> > (d_AraNöronlar, d_ÇıkışNöronları, d_Biaslar, İkinciKatmanAğırlıkları, ToplamAraNöronSayısı, ToplamÇıkışNöronuSayısı);
		cudaDeviceSynchronize();
	}


	// Formül: (Tahmin - Gerçek) * Sigmoid Türevi
	__global__ void IlkDeltaHesaplaKernel(float* d_ÇıkışNöronları, float* d_Hedefler, float* d_İlkDelta, int ÇıkışSayısı) {
		int BakılanNöron = blockIdx.x * blockDim.x + threadIdx.x;
		if (BakılanNöron >= ÇıkışSayısı) return;

		float t = d_ÇıkışNöronları[BakılanNöron]; // Tahmin
		float h = d_Hedefler[BakılanNöron];
		// Gerçek (Hedef)
		// Sigmoid Türevi: t * (1 - t)
		// Hata Türevi (MSE): (t - h)
		// Delta = (t - h) * t * (1.0f - t)
		d_İlkDelta[BakılanNöron] = (t - h);
	}

	__global__ void IkinciDeltaHesaplaKernel(float* d_AraNöronlar, float* d_İlkDelta, float* d_İkinciKatmanAğırlıkları, float* d_İkinciDelta, int AraNöronSayısı, int ÇıkışNöronuSayısı) {
		int BakılanNöron = blockIdx.x * blockDim.x + threadIdx.x;
		if (BakılanNöron >= AraNöronSayısı) { return; }
		float hataToplamı = 0.0f;
		for (int i = 0; i < ÇıkışNöronuSayısı; i++) {
			hataToplamı += d_İkinciKatmanAğırlıkları[AraNöronSayısı * i + BakılanNöron] * d_İlkDelta[i];
		}
		// ReLU türevinde eğer nöron sönmüşse (değeri 0 veya negatifse) hatayı iletmez (Türev 0)
		// Eğer nöron aktifse hatayı olduğu gibi iletir (Türev 1)

		if (d_AraNöronlar[BakılanNöron] > 0.0f) {
			d_İkinciDelta[BakılanNöron] = hataToplamı; // Hatayı olduğu gibi geçir
		}
		else {
			d_İkinciDelta[BakılanNöron] = hataToplamı * 0.1f; // Sızıntı tüneli
		}
	}

	//İkinci Katman (Ara -> Çıkış) Ağırlıklarını Güncelle
	__global__ void IkinciAgirliklariGuncelleKernel(float* d_İkinciKatmanAğırlıkları, float* d_AraNoronlar, float* d_İlkDelta, int AraSayısı, int ÇıkışSayısı, float ÖğrenmeOranı) {
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int ToplamAğırlık = AraSayısı * ÇıkışSayısı;
		if (id >= ToplamAğırlık) return;

		int ÇıkışNo = id / AraSayısı; // Hangi çıkışa gidiyor (tam kısmı)
		int AraNo = id % AraSayısı; // Hangi ara nörondan geliyor (bölümden kalan)

		// Gradient Descent: yeni_ağırlık = eski_ağırlık - (ÖğrenmeOranı * Delta * Giriş)
		d_İkinciKatmanAğırlıkları[id] -= ÖğrenmeOranı * d_İlkDelta[ÇıkışNo] * d_AraNoronlar[AraNo];
	}

	//İlk Katman (Giriş -> Ara) Ağırlıklarını Güncelle
	__global__ void IlkAgirliklariGuncelleKernel(float* d_IlkKatmanAğırlıkları, float* d_GirişNöronları, float* d_İkinciDelta, int GirişSayısı, int AraSayısı, float ÖğrenmeOranı) {
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		int ToplamAgirlik = GirişSayısı * AraSayısı;
		if (id >= ToplamAgirlik) return;

		int AraNo = id / GirişSayısı;
		int GirisNo = id % GirişSayısı;

		d_IlkKatmanAğırlıkları[id] -= ÖğrenmeOranı * d_İkinciDelta[AraNo] * d_GirişNöronları[GirisNo];
	}

	__global__ void BiasGuncelle(float* bias, float* delta, int n, float lr) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n) {
			bias[i] -= lr * delta[i];
		}
	}

	void HataHesapla(float* d_ÇıkışNöronları, float* d_AraNöronlar, float* d_Hedefler, float* d_İlkDelta, float* d_İkinciDelta, float* d_İkinciKatmanAğırlıkları, int ÇıkışNöronuSayısı, int AraNöronSayısı) {
		int blockSize = 256;
		int Blok1Sayısı = (ÇıkışNöronuSayısı + blockSize - 1) / blockSize;
		int Blok2Sayısı = (AraNöronSayısı + blockSize - 1) / blockSize;
		cudaDeviceSynchronize();
		IlkDeltaHesaplaKernel << <Blok1Sayısı, blockSize >> > (d_ÇıkışNöronları, d_Hedefler, d_İlkDelta, ÇıkışNöronuSayısı);
		cudaDeviceSynchronize();
		IkinciDeltaHesaplaKernel << <Blok2Sayısı, blockSize >> > (d_AraNöronlar, d_İlkDelta, d_İkinciKatmanAğırlıkları, d_İkinciDelta, AraNöronSayısı, ÇıkışNöronuSayısı);
		cudaDeviceSynchronize();
	}

	void KatmanAğırlıklarınıGüncelle(float* d_AraNöronlar, float* d_GirişNöronları, float* d_İlkKatmanAğırlıkları, float* d_İkinciKatmanAğırlıkları, float* d_İlkDelta, float* d_İkinciDelta, float* d_Bias1, float* d_Bias2, int AraNöronSayısı, int ÇıkışNöronuSayısı, int GirişNöronuSayısı, float ÖğrenmeOranı) {
		int İlkBiasSayısı = AraNöronSayısı;
		int İkinciBiasSayısı = ÇıkışNöronuSayısı;
		int blockSize = 256;
		int Blok1Sayısı = (AraNöronSayısı * ÇıkışNöronuSayısı + blockSize - 1) / blockSize;
		int Blok2Sayısı = (GirişNöronuSayısı * AraNöronSayısı + blockSize - 1) / blockSize;
		int blok3Sayısı = (İlkBiasSayısı + blockSize - 1) / blockSize;
		int blok4Sayısı = (İkinciBiasSayısı + blockSize - 1) / blockSize;
		cudaDeviceSynchronize();
		IkinciAgirliklariGuncelleKernel << <Blok1Sayısı, blockSize >> > (d_İkinciKatmanAğırlıkları, d_AraNöronlar, d_İlkDelta, AraNöronSayısı, ÇıkışNöronuSayısı, ÖğrenmeOranı);
		cudaDeviceSynchronize();
		IlkAgirliklariGuncelleKernel << <Blok2Sayısı, blockSize >> > (d_İlkKatmanAğırlıkları, d_GirişNöronları, d_İkinciDelta, GirişNöronuSayısı, AraNöronSayısı, ÖğrenmeOranı);
		cudaDeviceSynchronize();
		BiasGuncelle << <blok3Sayısı, blockSize >> > (d_Bias1, d_İkinciDelta, İlkBiasSayısı, ÖğrenmeOranı);
		BiasGuncelle << <blok4Sayısı, blockSize >> > (d_Bias2, d_İlkDelta, İkinciBiasSayısı, ÖğrenmeOranı);
		cudaDeviceSynchronize();
	}




	void turSuresiOlc(const char* etiket = "") {
		// static değişkenler program boyunca değerini korur
		static auto oncekiZaman = chrono::high_resolution_clock::now();
		static bool ilkCalisma = true;

		// Şu anki zamanı al
		auto suAn = chrono::high_resolution_clock::now();
		if (ilkCalisma) {
			cout << "--- Sayac Baslatildi ---" << endl;
			ilkCalisma = false;
		}
		else {
			// Aradaki farkı milisaniye (double) cinsinden hesapla
			chrono::duration<double, milli> gecenSure = suAn - oncekiZaman;
			cout << "Gecen Sure (" << etiket << "): "
				<< gecenSure.count() << " ms";
			if (gecenSure.count() > 0) {
				cout << " | " << (1000.0 / gecenSure.count()) << " FPS";
			}
			cout << endl;
		}
		// "Önceki zamanı" şu anki zaman olarak güncelle
		oncekiZaman = suAn;
	}


	int main(int argc, char* argv[])
	{
	// argc: Programın adı dahil kaç parametre girildiği.
	// argv[0]: Programın kendi adı (örn: sinir_agi.exe)
	// argv[1]: İlk parametre (Dosya Adı)
	// argv[2]: İkinci parametre (Mod)

	// 1. Parametre Kontrolü
		if (argc < 3) {
			cout << "Exe dosyasini dogrudan acmayiniz cmd ye cd (dosanin bulundugu konum) yazip enter dedikten sonra Kullanim de belirtilen sekilde parametreleri giriniz." << endl;
			cout << "Eksik parametre girdiniz!" << endl;
			cout << "Kullanim: program.exe <DosyaAdi.bmp> <Mod(1=Egitim, 2=Test)>" << endl;
			cout << R"(Orn: cd C:\Users\ahmet\OneDrive\Masaustu\)" << endl;
			cout << "Cuda.exe gorsel.bmp 2"<< endl;
			system("pause");
			return 0;
		}
		// 2. Parametreleri Değişkenlere Ata
		string GirilenDosyaAdi = argv[1]; // char*'dan string'e çevirir
		int kullanımodu = atoi(argv[2]);   //1 eğitim 2 test

		// 3. Mod Kontrolü
		if (kullanımodu != 1 && kullanımodu != 2) {
			cout << "Hatali Mod! Sadece 1 (Egitim) veya 2 (Test) girebilirsiniz." << endl;
			return 0;
		}

		cout << "Dosya: " << GirilenDosyaAdi << " | Mod: " << kullanımodu << " baslatiliyor..." << endl;

		int BelirlenenTurSayısı = 6000;
		if (kullanımodu==1) {
			vector<uint8_t>HeaderVerisiByte(60);
			int headerSize=0;
			int Genişlik = 0;
			int Yükseklik = 0;
			int padding = 0;
			ifstream görsel(GirilenDosyaAdi, ios::binary);
			if (!görsel) {
				cout << GirilenDosyaAdi <<" dosyası bulunamadı";
				return 0;
			}
			HeaderOku(HeaderVerisiByte, headerSize, Genişlik, Yükseklik, padding, görsel);
			int satirByteUzunlugu = (Genişlik * 3 + padding);//görseldeki her bir satırın byte uzunluğunu burda hesapladıkki işlemci yorulmasın

			vector <float> h_GirişNöronları(Genişlik*Yükseklik);
			vector <float> h_ÇıkışNöronları(10);
			vector <float> h_AraNöronlar(2 * (h_ÇıkışNöronları.size() + h_GirişNöronları.size()) / 3);
			int İlkKatmanAğırlıkSayısı = h_GirişNöronları.size() * h_AraNöronlar.size();
			int İkinciKatmanAğırlıkSayısı = h_AraNöronlar.size() * h_ÇıkışNöronları.size();
			int Bias1Sayısı = h_AraNöronlar.size();
			int Bias2Sayısı = h_ÇıkışNöronları.size();

			string İlkKatmanAğırlıklarıİsim = "İlkKatmanAğırlıkları(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream İlkDosyaVarmı(İlkKatmanAğırlıklarıİsim, ios::binary);
			string İkinciKatmanAğırlıklarıİsim = "İkinciKatmanAğırlıkları(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream İkinciDosyaVarmı(İkinciKatmanAğırlıklarıİsim, ios::binary);
			string Bias1İsim = "Bias1(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream Bias1DosyaVarmı(Bias1İsim, ios::binary);
			string Bias2İsim = "Bias2(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream Bias2DosyaVarmı(Bias2İsim, ios::binary);
			if (!İlkDosyaVarmı||!İkinciDosyaVarmı||!Bias1DosyaVarmı||!Bias2DosyaVarmı) {
				cout << "Girilen dosya buyuklugu icin gerekli egitim dosyalari bulunamadi program kapatiliyor...";
				return 0;
			}
			İlkDosyaVarmı.close();
			İkinciDosyaVarmı.close();
			Bias1DosyaVarmı.close();
			Bias2DosyaVarmı.close();
			ifstream Bias1Dosya(Bias1İsim, ios::binary);
			ifstream Bias2Dosya(Bias2İsim, ios::binary);
			ifstream İlkKatmanAğırlıklarıDosya(İlkKatmanAğırlıklarıİsim, ios::binary);
			ifstream İkinciKatmanAğırlıklarıDosya(İkinciKatmanAğırlıklarıİsim, ios::binary);
			// bu katman ağırlıklarını artık programımızın belleğine çekebiliriz
			vector <float> h_İlkKatmanAğırlıkları(İlkKatmanAğırlıkSayısı);
			vector <float> h_İkinciKatmanAğırlıkları(İkinciKatmanAğırlıkSayısı);
			vector <float> h_Bias1(Bias1Sayısı);
			vector <float> h_Bias2(Bias2Sayısı);
			vector<uint8_t>h_OrijinalGörsel(satirByteUzunlugu * Yükseklik);//h_OrijinalGörsel şeklinde yapılan isimlendirmedeki h_ bu değişkenin host(yani işlemci) tarafında olduğunu gösteriyor d_ olsaydı (device yani gpu) tarafında olurdu
			görsel.seekg(headerSize, ios::beg);
			görsel.read((char*)h_OrijinalGörsel.data(), h_OrijinalGörsel.size());
			görsel.close();

			İlkKatmanAğırlıklarıDosya.read((char*)h_İlkKatmanAğırlıkları.data(), İlkKatmanAğırlıkSayısı * sizeof(float));
			İkinciKatmanAğırlıklarıDosya.read((char*)h_İkinciKatmanAğırlıkları.data(), İkinciKatmanAğırlıkSayısı * sizeof(float));
			Bias1Dosya.read((char*)h_Bias1.data(), Bias1Sayısı * sizeof(float));
			Bias2Dosya.read((char*)h_Bias2.data(), Bias2Sayısı * sizeof(float));
			İlkKatmanAğırlıklarıDosya.close();
			İkinciKatmanAğırlıklarıDosya.close();
			Bias1Dosya.close();
			Bias2Dosya.close();
			//Artık Giriş Nöronları, 1.KatmanAğırlıkları ve 2.Katman ağırlıkları programımızın CPU belleğinde 
			//device değişkenlerimizi tanımlayalım
			fill(h_GirişNöronları.begin(), h_GirişNöronları.end(), 0.0f);
			float* d_GirişNöronları;
			float* d_AraNöronlar;
			float* d_ÇıkışNöronları;
			float* d_İlkKatmanAğırlıkları;
			float* d_İkinciKatmanAğırlıkları;
			float* d_Bias1;
			float* d_Bias2;
			//VRAM'da yer ayıralım
			cudaMalloc((void**)&d_GirişNöronları, h_GirişNöronları.size() * sizeof(float));
			cudaMalloc((void**)&d_AraNöronlar, h_AraNöronlar.size() * sizeof(float));
			cudaMalloc((void**)&d_ÇıkışNöronları, h_ÇıkışNöronları.size() * sizeof(float));
			cudaMalloc((void**)&d_İlkKatmanAğırlıkları, İlkKatmanAğırlıkSayısı * sizeof(float));
			cudaMalloc((void**)&d_İkinciKatmanAğırlıkları, İkinciKatmanAğırlıkSayısı * sizeof(float));
			cudaMalloc((void**)&d_Bias1, h_Bias1.size() * sizeof(float));
			cudaMalloc((void**)&d_Bias2, h_Bias2.size() * sizeof(float));
			//ağırlıkları VRAM'a kopyalayalım
			cudaMemcpy(d_GirişNöronları, h_GirişNöronları.data(), h_GirişNöronları.size() * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_İlkKatmanAğırlıkları, h_İlkKatmanAğırlıkları.data(), İlkKatmanAğırlıkSayısı * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_İkinciKatmanAğırlıkları, h_İkinciKatmanAğırlıkları.data(), İkinciKatmanAğırlıkSayısı * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Bias1, h_Bias1.data(), h_Bias1.size() * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Bias2, h_Bias2.data(), h_Bias2.size() * sizeof(float), cudaMemcpyHostToDevice);
			//GPU VRAM'ında Bellek Ayırdık
			// DÜZELTME: Veri işleme döngüsü
			int nöronSayacı = 0;
			for (int y = Yükseklik - 1; y >= 0; y--) {
				for (int x = 0; x < Genişlik; x++) {
					int idx = (y * satirByteUzunlugu) + (x * 3);

					uint8_t mavi = h_OrijinalGörsel[idx];
					uint8_t yeşil = h_OrijinalGörsel[idx + 1];
					uint8_t kırmızı = h_OrijinalGörsel[idx + 2];
					// Gri tonlama ve ters çevirme
					float parlaklik = (float)(mavi + yeşil + kırmızı) / 3.0f;
					h_GirişNöronları[nöronSayacı] = 1.0f - (parlaklik / 255.0f);

					nöronSayacı++;
				}
			}
			cudaMemcpy(d_GirişNöronları, h_GirişNöronları.data(), h_GirişNöronları.size() * sizeof(float), cudaMemcpyHostToDevice);

			turSuresiOlc();
			İlkİleriYayılım(d_GirişNöronları, d_AraNöronlar, d_Bias1, d_İlkKatmanAğırlıkları, h_GirişNöronları.size(), h_AraNöronlar.size());
			İkinciİleriYayılım(d_AraNöronlar, d_ÇıkışNöronları, d_Bias2, d_İkinciKatmanAğırlıkları, h_AraNöronlar.size(), h_ÇıkışNöronları.size());
			cudaMemcpy(h_ÇıkışNöronları.data(), d_ÇıkışNöronları, h_ÇıkışNöronları.size() * sizeof(float), cudaMemcpyDeviceToHost);
			float temp = 0;
			int temp1 = 0;
			for (int i = 0; i < h_ÇıkışNöronları.size(); i++) {
				if (h_ÇıkışNöronları[i] > temp) {
					temp = h_ÇıkışNöronları[i];
					temp1 = i;
				}
			}
			cout << "En aktif noron: " << temp1 << endl;
			cout << "Degeri: " << temp << endl;
			cudaDeviceSynchronize();
			turSuresiOlc("Kod Bitti");
			cudaFree(d_GirişNöronları);
			cudaFree(d_AraNöronlar);
			cudaFree(d_ÇıkışNöronları);
			cudaFree(d_İlkKatmanAğırlıkları);
			cudaFree(d_İkinciKatmanAğırlıkları);
			cudaFree(d_Bias1);
			cudaFree(d_Bias2);
			system("pause");
			return 0;
		}
			ifstream ilkGörsel("Tur(0)Sayi(0).bmp",ios::binary);
			if (!ilkGörsel) {
				cout << "Header Verisi icin ilk gorsel okunamadi (Tur(0)Sayi(0).bmp)" << endl;
			}
			//Görselin açlıp açılamadığı kontrol edildi.
			vector<uint8_t>HeaderVerisiByte(60);
			int HeaderSize;
			int Genişlik;
			int Yükseklik;
			int padding;
			HeaderOku(HeaderVerisiByte, HeaderSize, Genişlik, Yükseklik, padding, ilkGörsel);//header verileri okundu artık görselin kendi bytelarını okuyabiliriz
			int satirByteUzunlugu = (Genişlik * 3 + padding);//görseldeki her bir satırın byte uzunluğunu burda hesapladıkki işlemci yorulmasın
			ilkGörsel.close();//ifstream ile işimiz bitti (temizlik) 
			vector <float> h_GirişNöronları(Genişlik* Yükseklik);
			vector <float> h_ÇıkışNöronları(10);
			vector <float> h_AraNöronlar(2 * (h_ÇıkışNöronları.size() + h_GirişNöronları.size()) / 3);
			int İlkKatmanAğırlıkSayısı = h_GirişNöronları.size() * h_AraNöronlar.size();
			int İkinciKatmanAğırlıkSayısı = h_AraNöronlar.size() * h_ÇıkışNöronları.size();
			int Bias1Sayısı = h_AraNöronlar.size();
			int Bias2Sayısı = h_ÇıkışNöronları.size();
			string İlkKatmanAğırlıklarıİsim = "İlkKatmanAğırlıkları(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream İlkDosyaVarmı(İlkKatmanAğırlıklarıİsim, ios::binary);
			string İkinciKatmanAğırlıklarıİsim = "İkinciKatmanAğırlıkları(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream İkinciDosyaVarmı(İkinciKatmanAğırlıklarıİsim, ios::binary);
			string Bias1İsim = "Bias1(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream Bias1DosyaVarmı(Bias1İsim, ios::binary);
			string Bias2İsim = "Bias2(" + to_string(Genişlik) + "x" + to_string(Yükseklik) + ").bin";
			ifstream Bias2DosyaVarmı(Bias2İsim, ios::binary);
			// Eğer Bu isimde dosya yoksa bu if bloğu ile o isimde dosya oluşturup içlerini rasgele katman ağırlıklarıyla dolduruyoruz
			if (!İlkDosyaVarmı || !İkinciDosyaVarmı || !Bias1DosyaVarmı || !Bias2DosyaVarmı) {
				cout << "Dosyalardan biri veya birkaci eksik yeni dosyalar olusturuluyor..." << endl;
				ofstream İlkRasgeleKatmanAğırlıkları(İlkKatmanAğırlıklarıİsim, ios::binary);
				ofstream İkinciRasgeleKatmanAğırlıkları(İkinciKatmanAğırlıklarıİsim, ios::binary);
				ofstream Bias1Oluştur(Bias1İsim, ios::binary);
				ofstream Bias2Oluştur(Bias2İsim, ios::binary);
				vector <float> Bias1BoşVeri(Bias1Sayısı);
				vector <float> Bias2BoşVeri(Bias2Sayısı);
				fill(Bias1BoşVeri.begin(), Bias1BoşVeri.end(), 0);
				fill(Bias2BoşVeri.begin(), Bias2BoşVeri.end(), 0);
				// --- RASTGELE SAYI ÜRETECİ AYARLARI ---
						// Random device ve Mersenne Twister motoru (rand()'dan çok daha iyidir)
				random_device rd;
				mt19937 gen(rd());
				// Ağırlıklar genelde -0.5 ile 0.5 arasında veya -1 ile 1 arasında başlatılır.
				uniform_real_distribution <float> Dağılım(-0.05f, 0.05f);
				vector<float> BirinciKatmanVerisi(İlkKatmanAğırlıkSayısı);
				for (int i = 0; i < İlkKatmanAğırlıkSayısı; i++) {
					BirinciKatmanVerisi[i] = Dağılım(gen); // Sayıları tek tek üretip dolduruyoruz
				}
				// sizeof(float) ile çarparak BYTE cinsinden yazıyoruz:
				İlkRasgeleKatmanAğırlıkları.write((char*)BirinciKatmanVerisi.data(), İlkKatmanAğırlıkSayısı * sizeof(float));

				// --- 2. KATMANI OLUŞTUR VE YAZ ---
				vector<float> İkinciKatmanVerisi(İkinciKatmanAğırlıkSayısı);
				for (int i = 0; i < İkinciKatmanAğırlıkSayısı; i++) {
					İkinciKatmanVerisi[i] = Dağılım(gen);
				}
				İkinciRasgeleKatmanAğırlıkları.write((char*)İkinciKatmanVerisi.data(), İkinciKatmanAğırlıkSayısı * sizeof(float));
				Bias1Oluştur.write((char*)Bias1BoşVeri.data(), Bias1Sayısı * sizeof(float));
				Bias2Oluştur.write((char*)Bias2BoşVeri.data(), Bias2Sayısı * sizeof(float));
				İlkRasgeleKatmanAğırlıkları.close();
				İkinciRasgeleKatmanAğırlıkları.close();
				Bias1Oluştur.close();
				Bias2Oluştur.close();
			}
			İlkDosyaVarmı.close();
			İkinciDosyaVarmı.close();
			Bias1DosyaVarmı.close();
			Bias2DosyaVarmı.close();
			ifstream Bias1Dosya(Bias1İsim, ios::binary);
			ifstream Bias2Dosya(Bias2İsim, ios::binary);
			ifstream İlkKatmanAğırlıklarıDosya(İlkKatmanAğırlıklarıİsim, ios::binary);
			ifstream İkinciKatmanAğırlıklarıDosya(İkinciKatmanAğırlıklarıİsim, ios::binary);
			// bu katman ağırlıklarını artık programımızın belleğine çekebiliriz
			vector <float> h_İlkKatmanAğırlıkları(İlkKatmanAğırlıkSayısı);
			vector <float> h_İkinciKatmanAğırlıkları(İkinciKatmanAğırlıkSayısı);
			vector <float> h_Bias1(Bias1Sayısı);
			vector <float> h_Bias2(Bias2Sayısı);
			İlkKatmanAğırlıklarıDosya.read((char*)h_İlkKatmanAğırlıkları.data(), İlkKatmanAğırlıkSayısı * sizeof(float));
			İkinciKatmanAğırlıklarıDosya.read((char*)h_İkinciKatmanAğırlıkları.data(), İkinciKatmanAğırlıkSayısı * sizeof(float));
			Bias1Dosya.read((char*)h_Bias1.data(), Bias1Sayısı * sizeof(float));
			Bias2Dosya.read((char*)h_Bias2.data(), Bias2Sayısı * sizeof(float));
			İlkKatmanAğırlıklarıDosya.close();
			İkinciKatmanAğırlıklarıDosya.close();
			Bias1Dosya.close();
			Bias2Dosya.close();
			//Artık Giriş Nöronları, 1.KatmanAğırlıkları ve 2.Katman ağırlıkları programımızın CPU belleğinde 
			//device değişkenlerimizi tanımlayalım
			float* d_GirişNöronları;
			float* d_AraNöronlar;
			float* d_ÇıkışNöronları;
			float* d_İlkKatmanAğırlıkları;
			float* d_İkinciKatmanAğırlıkları;
			float* d_Bias1;
			float* d_Bias2;
			//VRAM'da yer ayıralım
			cudaMalloc((void**)&d_GirişNöronları, h_GirişNöronları.size() * sizeof(float));
			cudaMalloc((void**)&d_AraNöronlar, h_AraNöronlar.size() * sizeof(float));
			cudaMalloc((void**)&d_ÇıkışNöronları, h_ÇıkışNöronları.size() * sizeof(float));
			cudaMalloc((void**)&d_İlkKatmanAğırlıkları, İlkKatmanAğırlıkSayısı * sizeof(float));
			cudaMalloc((void**)&d_İkinciKatmanAğırlıkları, İkinciKatmanAğırlıkSayısı * sizeof(float));
			cudaMalloc((void**)&d_Bias1, h_Bias1.size() * sizeof(float));
			cudaMalloc((void**)&d_Bias2, h_Bias2.size() * sizeof(float));
			cudaDeviceSynchronize();
			//ağırlıkları VRAM'a kopyalayalım
			cudaMemcpy(d_İlkKatmanAğırlıkları, h_İlkKatmanAğırlıkları.data(), İlkKatmanAğırlıkSayısı * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_İkinciKatmanAğırlıkları, h_İkinciKatmanAğırlıkları.data(), İkinciKatmanAğırlıkSayısı * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Bias1, h_Bias1.data(), h_Bias1.size() * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Bias2, h_Bias2.data(), h_Bias2.size() * sizeof(float), cudaMemcpyHostToDevice);
			//GPU VRAM'ında Bellek Ayırdık
			float* d_Hedefler;
			vector <float> h_Hedefler(h_ÇıkışNöronları.size());
			fill(h_Hedefler.begin(), h_Hedefler.end(), 0);
			float* d_İlkDelta;
			float* d_İkinciDelta;
			float öğrenmeOranı = 0.01f; // Learning Rate (Çok büyük olursa öğrenemez, küçük başlamalıyız)
			cudaDeviceSynchronize();
			cudaMalloc((void**)&d_İlkDelta, h_ÇıkışNöronları.size() * sizeof(float));
			cudaMalloc((void**)&d_İkinciDelta, h_AraNöronlar.size() * sizeof(float));
			cudaMalloc((void**)&d_Hedefler, h_ÇıkışNöronları.size() * sizeof(float));
			cudaDeviceSynchronize();
			vector<uint8_t>h_OrijinalGörsel(satirByteUzunlugu * Yükseklik);//h_OrijinalGörsel şeklinde yapılan isimlendirmedeki h_ bu değişkenin host(yani işlemci) tarafında olduğunu gösteriyor d_ olsaydı (device yani gpu) tarafında olurdu

		for (int Tur = 0; Tur < BelirlenenTurSayısı; Tur++) {
			for (int BakılanSayı = 0; BakılanSayı < 10; BakılanSayı++) {
				fill(h_GirişNöronları.begin(), h_GirişNöronları.end(), 0.0f);
				string BakılanDosya = "Tur(" + to_string(Tur) + ")Sayi(" + to_string(BakılanSayı) + ").bmp";
				ifstream görsel(BakılanDosya, ios::binary);// Görseli byte byte oku
				if (!görsel) {
					cout << BakılanDosya << " Bulunamadi Sonraki Dosyaya Geciliyor" << endl;
					görsel.close();
					continue;
				}
				//Görselin açlıp açılamadığı kontrol edildi.
				görsel.seekg(HeaderSize, ios::beg);//başlangıçtan itibaren 54 byte ileri git (header 54. byte te bitiyor)
				görsel.read((char*)h_OrijinalGörsel.data(), h_OrijinalGörsel.size());//görsel Byte ları okunuyor
				görsel.close();//ifstream ile işimiz bitti (temizlik) 

				int nöronSayacı = 0;
				for (int y = Yükseklik - 1; y >= 0; y--) {
					for (int x = 0; x < Genişlik; x++) {
						int idx = (y * satirByteUzunlugu) + (x * 3);

						uint8_t mavi = h_OrijinalGörsel[idx];
						uint8_t yeşil = h_OrijinalGörsel[idx + 1];
						uint8_t kırmızı = h_OrijinalGörsel[idx + 2];
						// 1. Ortalama Parlaklığı Al (0=Siyah, 255=Beyaz)
						float parlaklik = (float)(mavi + yeşil + kırmızı) / 3.0f;

						// 2. Matematiksel Ters Çevirme (Inverted Grayscale)
						// Bu formül hem gri tonları korur hem de siyah yazıyı öne çıkarır.
						// Beyaz(255) -> 1.0 - 1.0 = 0.0 (Zemin yok olur)
						// Siyah(0)   -> 1.0 - 0.0 = 1.0 (Yazı parlar)
						h_GirişNöronları[nöronSayacı] = 1.0f - (parlaklik / 255.0f);
						nöronSayacı++;
					}
				}
				turSuresiOlc();
				cudaMemcpy(d_GirişNöronları, h_GirişNöronları.data(), h_GirişNöronları.size() * sizeof(float), cudaMemcpyHostToDevice);
				İlkİleriYayılım(d_GirişNöronları, d_AraNöronlar, d_Bias1, d_İlkKatmanAğırlıkları, h_GirişNöronları.size(), h_AraNöronlar.size());
				İkinciİleriYayılım(d_AraNöronlar, d_ÇıkışNöronları, d_Bias2, d_İkinciKatmanAğırlıkları, h_AraNöronlar.size(), h_ÇıkışNöronları.size());
				cudaMemcpy(h_ÇıkışNöronları.data(), d_ÇıkışNöronları, h_ÇıkışNöronları.size() * sizeof(float), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				//Geri Yayılım için hazırlık aşaması
				int doğruSonuç = BakılanSayı;
				float temp = 0;
				int temp1 = 0;
				for (int i = 0; i < h_ÇıkışNöronları.size(); i++) {
					if (h_ÇıkışNöronları[i] > temp) {
						temp = h_ÇıkışNöronları[i];
						temp1 = i;
					}
				}
				fill(h_Hedefler.begin(), h_Hedefler.end(), 0.0f);
				h_Hedefler[doğruSonuç] = 1.0f;
				cudaMemcpy(d_Hedefler, h_Hedefler.data(), h_Hedefler.size() * sizeof(float), cudaMemcpyHostToDevice);
				HataHesapla(d_ÇıkışNöronları, d_AraNöronlar, d_Hedefler, d_İlkDelta, d_İkinciDelta, d_İkinciKatmanAğırlıkları, h_ÇıkışNöronları.size(), h_AraNöronlar.size());
				KatmanAğırlıklarınıGüncelle(d_AraNöronlar, d_GirişNöronları, d_İlkKatmanAğırlıkları, d_İkinciKatmanAğırlıkları, d_İlkDelta, d_İkinciDelta, d_Bias1, d_Bias2, h_AraNöronlar.size(), h_ÇıkışNöronları.size(), h_GirişNöronları.size(), öğrenmeOranı);
				float toplamHata = 0.0f;
				for (int i = 0; i < 10; i++) {
					float fark = h_ÇıkışNöronları[i] - h_Hedefler[i];
					toplamHata += fark * fark;
				}

				// Ekrana yazdırma
				cout << "Dosya: " << BakılanDosya
					<< " | Tahmin: " << temp1
					<< " | Guven: " << temp
					<< " | Hata: " << toplamHata << endl;
			}
		}
		cudaDeviceSynchronize();
		cudaMemcpy(h_İlkKatmanAğırlıkları.data(), d_İlkKatmanAğırlıkları, h_İlkKatmanAğırlıkları.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_İkinciKatmanAğırlıkları.data(), d_İkinciKatmanAğırlıkları, h_İkinciKatmanAğırlıkları.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Bias1.data(), d_Bias1, h_Bias1.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Bias2.data(), d_Bias2, h_Bias2.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//cuda temizliği
		cudaFree(d_GirişNöronları);
		cudaFree(d_AraNöronlar);
		cudaFree(d_ÇıkışNöronları);
		cudaFree(d_İlkKatmanAğırlıkları);
		cudaFree(d_İkinciKatmanAğırlıkları);
		cudaFree(d_Bias1);
		cudaFree(d_Bias2);
		cudaFree(d_İlkDelta);
		cudaFree(d_İkinciDelta);
		cudaFree(d_Hedefler);
		//son olarak yeni katman ağırlıklarını ve Biasları dosyalara yazalım
		ofstream İlkKatmanAğırlıklarıYaz(İlkKatmanAğırlıklarıİsim, ios::binary);
		ofstream İkinciKatmanAğırlıklarıYaz(İkinciKatmanAğırlıklarıİsim, ios::binary);
		ofstream Bias1Yaz(Bias1İsim, ios::binary);
		ofstream Bias2Yaz(Bias2İsim, ios::binary);
		İlkKatmanAğırlıklarıYaz.write((char*)h_İlkKatmanAğırlıkları.data(), h_İlkKatmanAğırlıkları.size() * sizeof(float));
		İkinciKatmanAğırlıklarıYaz.write((char*)h_İkinciKatmanAğırlıkları.data(), h_İkinciKatmanAğırlıkları.size() * sizeof(float));
		Bias1Yaz.write((char*)h_Bias1.data(), h_Bias1.size() * sizeof(float));
		Bias2Yaz.write((char*)h_Bias2.data(), h_Bias2.size() * sizeof(float));
		İlkKatmanAğırlıklarıYaz.close();
		İkinciKatmanAğırlıklarıYaz.close();
		Bias1Yaz.close();
		Bias2Yaz.close();
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		system("pause");
	}