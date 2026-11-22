# Hệ màu Harmony Auralis

## Kết quả rà soát

Bốn màu tham chiếu đã được khai báo trong `theme.css` và `colors.ts`, đồng thời nhiều thành phần thư viện, Songscription và đăng nhập đã dùng biến CSS. Tuy nhiên, các file CSS chỉ nạp theme bằng `@import`; trong bản Expo/Metro được kiểm tra, các khai báo theme không xuất hiện trong stylesheet được gửi đến trình duyệt. Các biến như `--bg`, `--text`, `--accent` thực tế rỗng và nhiều thuộc tính màu trở về mặc định.

Piano còn ít nhất 60 giá trị màu viết trực tiếp trong CSS đang sử dụng; khuông nhạc SVG và icon native cũng có màu riêng. Các màu native/canvas và CSS được khai báo lặp, chưa có kiểm tra đồng bộ. Một số màu chữ phụ chưa đủ tương phản khi đặt trên nền lavender.

Theme hiện được import trực tiếp tại ba đầu vào DOM: `ExpoStudio`, `AuthDialog`, `PianoKeyboard`. Mọi màn hình sử dụng các đầu vào này, kể cả WebView trên native, nhận cùng một bộ biến màu.

## Phân vai màu

| Vai trò | Màu hoặc token | Cách sử dụng |
| --- | --- | --- |
| Lavender gốc | `#AFA2BA` | Minh họa, ảnh bìa, sắc độ nốt tay trái; không dùng làm chữ nhỏ trên nền sáng |
| Hồng khói gốc | `#CE96AD` | Chi tiết trang trí, ảnh bìa, nền hồng được pha loãng |
| San hô gốc | `#FF9BA1` | Nốt trực tiếp, từ đang hát được tô nền nhẹ; tránh phủ nền màn hình |
| Đỏ gạch gốc | `#BB6062` | Nốt tay phải, chi tiết âm nhạc |
| Thao tác chính | `primary`, `primaryHover` | Biến thể đỏ gạch đậm cho tạo, lưu, đăng nhập và nút phát; chữ trắng |
| Lựa chọn | `selection-bg/text/border` | Lavender nhạt với chữ tím đậm cho điều hướng, tab, bộ lọc và công cụ đang chọn |
| Đang phát | `playing-bg/text`, `primary/onPrimary` | Nền câu hồng nhạt; từ đang hát dùng nền đỏ gạch đậm, chữ trắng; từ đã hát giữ chữ đỏ gạch |
| Bề mặt | `background`, `surface`, `surfaceMuted` | Gần trắng, trắng, xám lavender rất nhạt; chiếm phần lớn diện tích |
| Chữ | `text`, `textMuted`, `textSubtle` | Tím than và tím xám đậm, đọc được trên các nền sáng |
| Trường nhập | `control-border`, `focus-ring` | Viền đủ rõ và vòng focus tím; không dùng viền phân cách nhạt thay viền trường nhập |
| Phím và khuông nhạc | `key-*`, `score-*` | Phím trắng–đen, giấy trắng, mực tối; màu chỉ dành cho trạng thái đang chơi |
| Phản hồi | `success`, `warning`, `danger` | Xanh trầm, nâu hổ phách, đỏ lỗi với nền tương ứng; luôn đi kèm nội dung/trạng thái |

Giữ các vùng lớn trung tính. Dùng đỏ gạch đậm cho hành động nổi bật nhất trong một nhóm; các nút phụ dùng nền trắng hoặc trung tính. Gradient chỉ dành cho vùng thương hiệu và ảnh bìa. Không đổi màu mọi icon, văn bản hay nút sang bốn màu gốc.

Màu hai tay phải thống nhất giữa phím đang nhấn, waterfall và chú giải. `leftHand/rightHand` dành cho hình vẽ; dùng `secondary/primary` khi cần chữ nhỏ trên nền sáng.

## Nguồn và cách bảo trì

- Sửa giá trị tại `src/theme/tokens.json`: `palette` chứa màu tham chiếu, `colors` chứa màu theo vai trò, `css` ánh xạ các vai trò sang tên biến CSS.
- `src/theme/colors.ts` đọc cùng JSON cho native và canvas.
- Chạy `npm run theme:generate` để tạo lại `src/styles/theme.css`; không sửa file được tạo bằng tay.
- Chạy `npm run theme:check` để kiểm tra file sinh ra, tham chiếu biến, đường nạp theme, màu viết trực tiếp và độ tương phản. Build web tự chạy kiểm tra này.
- Màu chữ được kiểm tra tối thiểu 4.5:1 trên các bề mặt dùng thực tế; viền trường nhập tối thiểu 3:1. Bản in vẫn dùng giấy trắng và mực đen.
- `app.json` giữ màu tĩnh cho splash/adaptive icon; kiểm tra theme sẽ phát hiện nếu chúng lệch token. Nền ảnh xanh mẫu của adaptive icon đã được gỡ khỏi cấu hình.

Các file `src/features/user/piano/pages/*.html` và `styles/main.css` là mẫu upstream cũ, không được import hay xuất thành route của ứng dụng. Chúng được giữ nguyên và nằm ngoài kiểm tra UI đang chạy. Nếu đưa chúng vào sử dụng, cần chuyển chúng sang theme chung và bổ sung vào kiểm tra.

## Kiểm chứng ngày 21/11/2026

- `theme:check`: 53 cặp màu đạt yêu cầu; độ tương phản chữ thấp nhất trong các cặp kiểm tra là 4,60:1, kể cả màu tổng hợp trên phím đen đang nhấn. CSS đang dùng không còn mã màu viết trực tiếp ngoài nguồn token được sinh ra.
- TypeScript và build web chạy thành công. Bản export có stylesheet theme riêng.
- Đã chụp và kiểm tra 13 màn hình/khung nhìn trên bản build: trang chủ, Piano, Projects, Collections, Songscription, hộp tạo Songscription và đăng nhập/đăng ký; gồm desktop, điện thoại rộng 375 px, tablet và màn hình ngang. Không có tràn ngang ở cấp trang hoặc ngoại lệ JavaScript trong các lượt kiểm tra này.
- 15 kiểm tra màu hiển thị đạt: nút chính, hover, focus, báo lỗi, màu đang chọn/đang phát, trạng thái thành công/cảnh báo và màu hai tay trên phím. Dòng transcript được kiểm tra bằng mẫu DOM với các class thực tế, không cần tạo hay sửa dữ liệu người dùng.
- Lint toàn bộ `src` còn 5 lỗi và 3 cảnh báo React hooks. Đã so với bản mã trước khi sửa màu và xác nhận không phát sinh chẩn đoán mới; các lỗi chức năng này nằm ngoài thay đổi bảng màu.

## Điều chỉnh Songscription sau rà soát sử dụng

Cột thông tin phiên/lịch sử phụ và nút thu gọn tương ứng đã được gỡ khỏi màn hình chi tiết; trình rà soát dùng toàn bộ chiều rộng còn lại. Phiên vẫn được quản lý qua thư viện Songscription.

Thanh phát dùng `--fill` theo thời gian thực của audio, bắt đầu từ 0% và kết thúc ở 100%. Lời được cập nhật từ cùng đồng hồ audio trong lúc phát, phân biệt từ sắp hát, đang hát và đã hát; tua ngược tính lại màu theo vị trí mới. Từ đang hát dùng nền đỏ gạch với chữ trắng để dễ thấy.

Nhóm điều khiển dùng `components/ui/Icon.tsx` cho lùi/tiến 5 giây, phát/tạm dừng và âm lượng. Piano và Songscription dùng chung `components/ui/VolumeControl.tsx`: bấm chỉ mở/đóng thanh kéo, không đổi mức âm lượng hoặc trạng thái tiếng. Không có nút mute riêng trong popup; kéo về 0 để im tiếng và kéo lên để tăng âm lượng. Popup nằm trên thanh công cụ Piano và dưới thanh phát Songscription, hỗ trợ chuột, cảm ứng, phím mũi tên và Escape.

Các kiểm tra hồi quy thời gian phát nằm tại `scripts/lyrics-playback.test.cjs`, chạy bằng `node --test scripts/lyrics-playback.test.cjs`.

Ở lượt rà soát trình phát ban đầu: 6 kiểm tra hồi quy đạt; 26 kiểm tra trình duyệt trên bản build với audio thật đạt, gồm phát/tạm dừng, tua hai chiều, đầu/cuối bài, highlight, âm lượng, bàn phím, mobile và tablet. Không có yêu cầu sửa dữ liệu người dùng trong các lượt kiểm tra.

Form tạo Songscription hiện theo cùng cấu trúc form Piano: panel trắng 450px, tiêu đề/mô tả, tab nguồn, vùng tải file, ô YouTube và nút chính toàn chiều rộng. Tab mặc định gộp âm thanh/YouTube; Projects là tab thứ hai. Các tùy chọn căn chỉnh lời, nhập TXT/LRC/SRT, ngôn ngữ, mô hình, tách giọng và tên tùy chọn nằm trong phần Cài đặt lời hát.

Ô YouTube chỉ có một viền focus ở khung bao, không vẽ thêm viền/outline trên input bên trong. Đã kiểm tra các nhánh dữ liệu của form bằng cách chặn yêu cầu tạo phiên trong trình duyệt; không tạo phiên thử trên backend. Kéo âm lượng bằng chuột đã được kiểm tra trên cả hai màn hình; 9 kiểm tra bản build bổ sung xác nhận giao diện mobile, focus YouTube và kéo cảm ứng cập nhật đúng gain Piano/volume audio Songscription. TypeScript, build, 6 kiểm tra hồi quy và lint riêng các file vừa sửa đều đạt; lint toàn dự án vẫn còn các lỗi React hooks đã ghi ở trên.
