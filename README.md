# Exercise M09W03 - AIO2024: VAE model

**AI tạo sinh (Generative AI)** đang trở thành một thuật ngữ phổ biến và xuất hiện rộng rãi trong nhiều lĩnh vực. Trên thực tế, AI tạo sinh không chỉ dừng lại ở việc
 xử lý và phân tích dữ liệu có sẵn mà còn có khả năng tạo ra dữ liệu mới từ đầu, bao gồm văn bản,
 hình ảnh, âm thanh và nhiều loại dữ liệu khác. Một trong những phương pháp cốt lõi của AI tạo sinh
 là **Variational AutoEncoder (VAE)**, một mô hình đóng vai trò quan trọng trong việc học biểu diễn
 dữ liệu và sinh dữ liệu mới. VAE không chỉ giúp chúng ta hiểu sâu hơn về quy trình tạo sinh mà còn là
 nền tảng của nhiều mô hình hiện đại, bao gồm các hệ thống tạo ảnh tiên tiến như Stable Diffusion.

## Autoencoder
Autoencoder (AE) là một mô hình học sâu được sử dụng để tái tạo dữ liệu, với mục tiêu nén thông
 tin đầu vào và sau đó phục hồi nó từ không gian ẩn.

![Autoencoder Architecture](/readme_image/autoencoder.png "AIO2024")

Mô hình AE có 3 phần chính: 
- **Encoder**: có nhiệm vụ chuyển đổi dữ liệu đầu vào $x \in \mathbb{R}^n$ thành một biểu diễn ở không gian ẩn $z \in \mathbb{R}^q$, với $q \ll n$. Quá trình này thực hiện việc nén dữ liệu, giảm chiều của dữ liệu gốc nhằm giữ lại những đặc trưng quan trọng nhất. Việc chuyển đổi này giúp mô hình tập trung vào các đặc trưng quan trọng của dữ liệu, loại bỏ những thông tin không cần thiết và tạo ra một biểu diễn gọn gàng hơn

- **Latent Space**: Không gian ẩn là một không gian có kích thước nhỏ hơn so với đầu vào của mô hình, nơi mô hình lưu trữ các đặc trưng cốt lõi của dữ liệu. Không gian này đóng vai trò quan trọng trong việc tạo ra một biểu diễn gọn gàng và hiệu quả của dữ liệu. Khi kích thước của không gian ẩn càng nhỏ, thông tin được lưu trữ trong latent space càng trở nên quan trọng và giúp tránh được vấn đề overfitting khi model phải chọn lọc các thông tin quan trọng để lưu trữ.

- **Decoder:** Phần decoder nhận đầu vào là các đặc trưng từ không gian ẩn và tái tạo lại dữ liệu đầu vào ban đầu. Mục tiêu của decoder là khôi phục lại dữ liệu gốc một cách chính xác nhất có thể, với sai số nhỏ nhất so với dữ liệu ban đầu. Quá trình này đóng vai trò quan trọng trong việc tái tạo thông tin từ biểu diễn ở không gian ẩn, đồng thời giúp mô hình hiểu rõ hơn về các đặc trưng đã được mã hóa trong quá trình encoder.

Tuy có nhiều lợi ích nhưng AE có một hạn chế lớn đó là việc không gian tiềm ẩn (latent space) không có cấu trúc xác suất, nghĩa là mô hình chỉ tối ưu hóa reconstruction loss mà không đảm bảo rằng không gian ẩn có tính liên tục và đầy đủ. Điều này có thể dẫn đến việc các điểm dữ liệu tương tự có thể bị ánh xạ đến các vị trí x nhau trong không gian ẩn. Thêm vào đó, điểm yếu này khiến AE không thể sinh ra dữ liệu mới một cách hợp lý, vì mô hình chỉ học cách sao chép dữ liệu đầu vào thay vì học một phân phối tổng quát của dữ liệu.

## Variational Autoencoder
Variational Autoencoder (VAE) là một mô hình học sâu sinh (generative deep learning model) được
 thiết kế để học các biểu diễn tiềm ẩn (latent representations) của dữ liệu một cách không giám sát.
 Không giống như các phương pháp giảm chiều đơn thuần, VAE học một không gian tiềm ẩn có cấu trúc
 và có thể lấy mẫu để tạo ra dữ liệu mới tương tự như dữ liệu huấn luyện. VAE kết hợp các kỹ thuật
 từ suy luận biến phân (variational inference) và mạng nơ-ron sâu (deep neural networks) để để tối ưu
 hóa không gian ẩn. Thay vì chỉ ánh xạ dữ liệu đầu vào thành một vector cố định, VAE ánh xạ dữ liệu
 vào một phân phối xác suất, cho phép sinh ra các dữ liệu mới từ không gian tiềm ẩn theo một cách có
 kiểm soát.

![VAE Architecture](/readme_image/vae.png "AIO2024")

**Cấu trúc của mô hình VAE:**
- **Encoder (Parametric Encoder):** Thay vì tạo ra một vector mã hóa đơn lẻ như AE, encoder
 của VAE tạo ra hai vector: vector trung bình (mean-$\mu$) và vector độ lệch chuẩn (deviation-$\sigma$). Mô
 hình sau đó sử dụng hai vector này để xác định phân phối Gaussian.

- **Latent space (Probabilistic Latent Space):** VAE không có một ”bottleneck”cố định như AE. Thay vào đó, mô hình ép buộc các biểu diễn trong không gian ẩn phải tuân theo một phân phối xác suất cụ thể. Điều này có tác dụng điều chuẩn (regularization), tạo cấu trúc và tính liên tục cho không gian ẩn.

- **Reparameterization Trick:** Để huấn luyện mô hình VAE bằng gradient descent, cần phải lấy mẫu từ phân phối xác suất được xác định bởi encoder. Tuy nhiên, thao tác lấy mẫu là không khả vi (non-differentiable). Để giải quyết vấn đề này, VAE sử dụng một kỹ thuật gọi là "reparameterization trick":
        $$z = \mu + \sigma \bigodot \epsilon$$
    Trong đó:\
        – $\mu$ là vector trung bình.\
        – $\sigma$ là vector độ lệch chuẩn.\
        – $\epsilon$ là biến ngẫu nhiên được lấy mẫu từ một phân phối cố định.

    Kỹ thuật này cho phép biểu diễn biến ngẫu nhiên z như một hàm khả vi của các tham số encoder và một biến ngẫu nhiên độc lập, sau z đó được truyền cho decoder.

- **Decoder:** Decoder nhận một mẫu từ không gian ẩn và tái tạo lại dữ liệu đầu vào ban đầu, tương tự như trong AE.

**Loss Function**: bao gồm 2 phần Reconstruction Loss và Regularization Loss

![Loss function ELBO](/readme_image/loss_vae.png "AIO2024")

- **Reconstruction Loss**: đo lường khả năng tái tạo dữ liệu x từ các mẫu z được lấy từ phân phối q(z|x).

- **Regularization Loss**: đo lường sự khác biệt giữa phân phối hậu nghiệm (posterior) gần đúng q(z|x) (do encoder ước tính) và phân phối tiên nghiệm (prior) p(z) (thường sử dụng Gaussian chuẩn). Regularization Loss đảm bảo rằng không gian tiềm ẩn có cấu trúc liên tục và có tính tổng quát cao, giúp VAE sinh ra các mẫu dữ liệu mới từ không gian này một cách hiệu quả. Đây là điểm khác biệt chính so với AE, vì AE không có thành phần điều chuẩn này, dẫn đến không gian tiềm ẩn của AE thường thiếu cấu trúc rõ ràng.