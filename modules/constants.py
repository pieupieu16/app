districts = [
    "Ba Đình", "Bắc Từ Liêm", "Cầu Giấy", "Đống Đa", "Hà Đông", "Hai Bà Trưng", 
    "Hoàn Kiếm", "Hoàng Mai", "Long Biên", "Nam Từ Liêm", "Tây Hồ", "Thanh Xuân",
    "Chương Mỹ", "Đan Phượng", "Đông Anh", "Gia Lâm", "Hoài Đức", "Mê Linh", 
    "Mỹ Đức", "Phú Xuyên", "Phúc Thọ", "Quốc Oai", "Sóc Sơn", "Thạch Thất", 
    "Thanh Oai", "Thanh Trì", "Thường Tín", "Thị xã Sơn Tây"
]
districts.sort()

# (Dữ liệu Wards map của bạn - giữ nguyên nhưng thu gọn hiển thị ở đây)
wards_map = {
    # CÁC QUẬN NỘI THÀNH VÀ LÂN CẬN
    "Ba Đình": ["Phường Cống Vị", "Phường Giảng Võ", "Phường Kim Mã", "Phường Liễu Giai", 
                "Phường Ngọc Hà", "Phường Ngọc Khánh", "Phường Phúc Xá", "Phường Quán Thánh", 
                "Phường Thành Công", "Phường Trúc Bạch", "Phường Vĩnh Phúc", "Phường Đội Cấn", "Phường Điện Biên"],
    
    "Hoàn Kiếm": ["Phường Chương Dương", "Phường Cửa Nam", "Phường Cửa Đông", "Phường Hàng Buồm", 
                "Phường Hàng Bài", "Phường Hàng Bông", "Phường Hàng Bạc", "Phường Hàng Bồ", 
                "Phường Hàng Gai", "Phường Hàng Mã", "Phường Hàng Trống", "Phường Hàng Đào", 
                "Phường Lý Thái Tổ", "Phường Phan Chu Trinh", "Phường Phúc Tân", "Phường Tràng Tiền", "Phường Đồng Xuân", "Phường Yết Kiêu"],
    
    "Hai Bà Trưng": ["Phường Bách Khoa", "Phường Bùi Thị Xuân", "Phường Bạch Mai", "Phường Bạch Đằng", 
                    "Phường Cầu Dền", "Phường Đồng Nhân", "Phường Đồng Tâm", "Phường Kim Liên", 
                    "Phường Lê Đại Hành", "Phường Minh Khai", "Phường Nguyễn Du", "Phường Ngô Thì Nhậm", 
                    "Phường Phạm Đình Hổ", "Phường Phố Huế", "Phường Quỳnh Lôi", "Phường Quỳnh Mai", 
                    "Phường Thanh Lương", "Phường Thanh Nhàn", "Phường Trương Định", "Phường Vĩnh Tuy", "Phường Đống Mác"],
    
    "Đống Đa": ["Phường Hàng Bột", "Phường Khâm Thiên", "Phường Khương Thượng", "Phường Kim Liên", 
                "Phường Láng Hạ", "Phường Láng Thượng", "Phường Nam Đồng", "Phường Nguyễn Trãi", 
                "Phường Ngã Tư Sở", "Phường Phương Liên", "Phường Phương Mai", "Phường Quốc Tử Giám", 
                "Phường Thịnh Quang", "Phường Thổ Quan", "Phường Trung Liệt", "Phường Trung Phụng", 
                "Phường Trung Tự", "Phường Văn Chương", "Phường Văn Miếu", "Phường Ô Chợ Dừa"],
    
    "Cầu Giấy": ["Phường Cầu Diễn", "Phường Dịch Vọng", "Phường Dịch Vọng Hậu", "Phường Mai Dịch", 
                "Phường Nghĩa Tân", "Phường Nghĩa Đô", "Phường Quan Hoa", "Phường Trung Hoà", "Phường Yên Hoà"],
    
    "Tây Hồ": ["Phường Bưởi", "Phường Nhật Tân", "Phường Quảng An", "Phường Thụy Khuê", 
            "Phường Tứ Liên", "Phường Xuân La", "Phường Yên Phụ"],

    "Thanh Xuân": ["Phường Hạ Đình", "Phường Khương Mai", "Phường Khương Trung", "Phường Khương Đình", 
                "Phường Kim Giang", "Phường Nhân Chính", "Phường Phương Liệt", "Phường Thanh Xuân Bắc", 
                "Phường Thanh Xuân Nam", "Phường Thanh Xuân Trung", "Phường Thượng Đình", "Phường Định Công"], # Định Công thường thuộc Hoàng Mai nhưng có thể liên quan
    
    "Hoàng Mai": ["Phường Giáp Bát", "Phường Hoàng Liệt", "Phường Hoàng Văn Thụ", "Phường Lĩnh Nam", 
                "Phường Mai Động", "Phường Thịnh Liệt", "Phường Trần Phú", "Phường Tân Mai", 
                "Phường Tương Mai", "Phường Vĩnh Hưng", "Phường Yên Sở", "Phường Đại Kim", 
                "Phường Định Công", "Phường Đồng Tâm", "Phường Vĩnh Tuy", "Phường Thanh Trì"], # (Phường, không phải Huyện)
    
    "Long Biên": ["Phường Bồ Đề", "Phường Cự Khối", "Phường Gia Thụy", "Phường Giang Biên", 
                "Phường Long Biên", "Phường Ngọc Lâm", "Phường Ngọc Thụy", "Phường Phúc Đồng", 
                "Phường Phúc Lợi", "Phường Phúc Tân", "Phường Phúc Xá", "Phường Sài Đồng", 
                "Phường Thạch Bàn", "Phường Thượng Thanh", "Phường Việt Hưng", "Phường Đức Giang"],

    "Bắc Từ Liêm": ["Phường Cầu Diễn", "Phường Cổ Nhuế 1", "Phường Cổ Nhuế 2", "Phường Liên Mạc", 
                    "Phường Minh Khai", "Phường Phú Diễn", "Phường Phúc Diễn", "Phường Thượng Cát", 
                    "Phường Thụy Phương", "Phường Tây Tựu", "Phường Xuân Tảo", "Phường Xuân Đỉnh", "Phường Đông Ngạc", "Phường Đức Thắng"],
    
    "Nam Từ Liêm": ["Phường Cầu Diễn", "Phường Mễ Trì", "Phường Mỹ Đình 1", "Phường Mỹ Đình 2", 
                    "Phường Phú Đô", "Phường Phương Canh", "Phường Trung Văn", "Phường Tây Mỗ", "Phường Xuân Phương", "Phường Đại Mỗ"],
    
    "Hà Đông": ["Phường Dương Nội", "Phường Hà Cầu", "Phường Kiến Hưng", "Phường La Khê", 
                "Phường Mộ Lao", "Phường Nguyễn Trãi", "Phường Phú La", "Phường Phú Lãm", 
                "Phường Phú Lương", "Phường Phú Thịnh", "Phường Phúc La", "Phường Quang Trung", 
                "Phường Vạn Phúc", "Phường Văn Quán", "Phường Yên Nghĩa", "Phường Đồng Mai"],
    
    # CÁC HUYỆN VÀ THỊ XÃ
    "Đông Anh": ["Thị trấn Đông Anh", "Xã Bắc Hồng", "Xã Dục Tú", "Xã Hải Bối", "Xã Kim Chung", 
                "Xã Kim Nỗ", "Xã Liên Hà", "Xã Mai Lâm", "Xã Nam Hồng", "Xã Nguyên Khê", 
                "Xã Tiên Dương", "Xã Uy Nỗ", "Xã Vân Nội", "Xã Võng La", "Xã Xuân Giang", 
                "Xã Xuân Nộn", "Xã Yên Thường", "Xã Đại Mạch", "Xã Đông Hội"],
    
    "Gia Lâm": ["Thị trấn Trâu Quỳ", "Thị trấn Yên Viên", "Xã Bát Tràng", "Xã Cổ Bi", "Xã Cự Khối", 
                "Xã Đa Tốn", "Xã Kiêu Kỵ", "Xã Ninh Hiệp", "Xã Phú Thị", "Xã Phù Đổng", 
                "Xã Trung Mầu", "Xã Yên Viên", "Xã Đông Dư", "Xã Đặng Xá", "Xã Đình Xuyên"],
    
    "Hoài Đức": ["Thị trấn Trạm Trôi", "Xã An Khánh", "Xã An Thượng", "Xã Cát Quế", "Xã Di Trạch", 
                "Xã Dương Liễu", "Xã Lại Yên", "Xã La Phù", "Xã Song Phương", "Xã Sơn Đồng", 
                "Xã Tiền Yên", "Xã Vân Canh", "Xã Vân Côn", "Xã Yên Sở", "Xã Đông La", "Xã Đức Thượng"],
    
    "Thanh Trì": ["Thị trấn Văn Điển", "Xã Duyên Hà", "Xã Duyên Thái", "Xã Hữu Hoà", "Xã Khánh Hà", 
                "Xã Liên Ninh", "Xã Ngọc Hồi", "Xã Ngũ Hiệp", "Xã Tả Thanh Oai", "Xã Tam Hiệp", 
                "Xã Tân Triều", "Xã Tứ Hiệp", "Xã Vĩnh Quỳnh", "Xã Văn Bình", "Xã Yên Mỹ", 
                "Xã Thanh Liệt"], # (Loại trừ các phường đã xếp vào Quận khác)
    
    "Thạch Thất": ["Thị trấn Liên Quan", "Xã Bình Phú", "Xã Bình Yên", "Xã Cẩm Quan", "Xã Cổ Đông", 
                "Xã Hạ Bằng", "Xã Hữu Bằng", "Xã Hương Ngải", "Xã Kim Quan", "Xã Lại Thượng", 
                "Xã Phú Kim", "Xã Phú Mãn", "Xã Phùng Xá", "Xã Tân Xã", "Xã Thạch Hoà", 
                "Xã Tiên Xuân", "Xã Yên Bình", "Xã Yên Trung", "Xã Canh Nậu", "Xã Đồng Trúc"],
    
    "Sóc Sơn": ["Thị trấn Sóc Sơn", "Xã Bắc Sơn", "Xã Hiền Ninh", "Xã Kim Lũ", "Xã Mai Đình", 
                "Xã Minh Phú", "Xã Minh Trí", "Xã Nam Sơn", "Xã Phù Linh", "Xã Phù Lỗ", 
                "Xã Quang Tiến", "Xã Tân Dân", "Xã Thanh Xuân", "Xã Tiên Dược", "Xã Trung Giã", 
                "Xã Việt Long", "Xã Xuân Giang", "Xã Xuân Thu"], # (Loại trừ các phường/xã đã xếp vào Quận khác)

    "Thường Tín": ["Thị trấn Thường Tín", "Xã Hà Hồi", "Xã Hiền Giang", "Xã Hòa Bình", "Xã Hồng Vân", 
                "Xã Khánh Hà", "Xã Lê Lợi", "Xã Liên Phương", "Xã Minh Cường", "Xã Nghiêm Xuyên", 
                "Xã Nhị Khê", "Xã Ninh Sở", "Xã Quất Động", "Xã Thắng Lợi", "Xã Thống Nhất", 
                "Xã Tiền Phong", "Xã Tô Hiệu", "Xã Tự Nhiên", "Xã Vạn Điểm", "Xã Văn Bình", "Xã Văn Phú"],

    "Chương Mỹ": ["Thị trấn Chúc Sơn", "Thị trấn Xuân Mai", "Xã Hợp Thanh", "Xã Nam Phương Tiến", "Xã Phụng Châu", 
                "Xã Thủy Xuân Tiên", "Xã Đông Phương Yên", "Xã Trung Hòa", "Xã Văn Võ", "Xã Đồng Lạc"],
    
    "Đan Phượng": ["Thị trấn Phùng", "Xã Đan Phượng", "Xã Đồng Tháp", "Xã Hạ Mỗ", "Xã Hồng Hà", 
                "Xã Liên Hà", "Xã Liên Hồng", "Xã Phương Đình", "Xã Song Phượng", "Xã Thọ An", 
                "Xã Thọ Xuân", "Xã Thượng Mỗ", "Xã Trung Châu"],
    
    "Phú Xuyên": ["Thị trấn Phú Xuyên", "Xã Bạch Hạ", "Xã Châu Can", "Xã Chuyên Mỹ", "Xã Đại Thắng", 
                "Xã Hồng Thái", "Xã Khai Thái", "Xã Minh Tân", "Xã Nam Phong", "Xã Nam Triều", 
                "Xã Phú Châu", "Xã Phú Túc", "Xã Phúc Tiến", "Xã Quang Lãng", "Xã Quang Trung", 
                "Xã Sơn Hà", "Xã Tân Dân", "Xã Tri Thủy", "Xã Tri Trung", "Xã Văn Hoàng", "Xã Vân Từ"],
    
    "Quốc Oai": ["Thị trấn Quốc Oai", "Xã Cấn Hữu", "Xã Cộng Hòa", "Xã Đại Thành", "Xã Đồng Quang", 
                "Xã Hòa Thạch", "Xã Liệp Tuyết", "Xã Ngọc Liệp", "Xã Ngọc Mỹ", "Xã Phú Cát", 
                "Xã Phú Mãn", "Xã Phượng Cách", "Xã Sài Sơn", "Xã Tuyết Nghĩa", "Xã Yên Sơn"],
    
    "Thị xã Sơn Tây": ["Phường Lê Lợi", "Phường Ngô Quyền", "Phường Phú Thịnh", "Phường Quang Trung", 
                    "Phường Sơn Lộc", "Phường Trung Hưng", "Phường Viên Sơn", "Phường Xuân Khanh", 
                    "Xã Cổ Đông", "Xã Đường Lâm", "Xã Kim Sơn", "Xã Sơn Đông", "Xã Thanh Mỹ", "Xã Xuân Sơn"],
    
    "Mê Linh": ["Thị trấn Quang Minh", "Xã Chu Phan", "Xã Đại Thịnh", "Xã Hoàng Kim", "Xã Kim Hoa", 
                "Xã Liên Mạc", "Xã Mê Linh", "Xã Tam Đồng", "Xã Thạch Đà", "Xã Tiền Phong", 
                "Xã Tráng Việt", "Xã Tự Lập", "Xã Văn Khê", "Xã Vạn Yên", "Xã Thanh Lâm"],
    
    "Phúc Thọ": ["Thị trấn Phúc Thọ", "Xã Cẩm Đình", "Xã Hát Môn", "Xã Hiệp Thuận", "Xã Liên Hiệp", 
                "Xã Long Xuyên", "Xã Ngọc Tảo", "Xã Phụng Thượng", "Xã Sen Chiểu", "Xã Tam Thuấn", 
                "Xã Thanh Đa", "Xã Thượng Cốc", "Xã Tích Giang", "Xã Vân Hà", "Xã Vân Nam", "Xã Võng Xuyên", "Xã Xuân Phú"],
    
    "Mỹ Đức": ["Thị trấn Đại Nghĩa", "Xã An Mỹ", "Xã An Phú", "Xã Bột Xuyên", "Xã Đại Hưng", 
            "Xã Đồng Tâm", "Xã Hồng Sơn", "Xã Hợp Thanh", "Xã Hợp Tiến", "Xã Hùng Tiến", 
            "Xã Hương Sơn", "Xã Lê Thanh", "Xã Mỹ Thành", "Xã Phù Lưu Tế", "Xã Phúc Lâm", 
            "Xã Thượng Lâm", "Xã Tuy Lai", "Xã Vạn Kim"],
    
    "Thanh Oai": ["Thị trấn Kim Bài", "Xã Bích Hòa", "Xã Cự Khê", "Xã Dân Hòa", "Xã Hồng Dương", 
                "Xã Kim An", "Xã Kim Thư", "Xã Liên Châu", "Xã Mỹ Hưng", "Xã Phương Trung", 
                "Xã Tam Hưng", "Xã Thanh Cao", "Xã Thanh Mai", "Xã Thanh Văn", "Xã Xuân Dương"],
    
    
}

SYSTEM_INSTRUCTION = """
BẠN LÀ: "Chuyên gia Chiến lược Đầu tư BĐS Hà Nội (Senior AI Analyst)". 
PHONG CÁCH: Chuyên nghiệp, nhạy bén với thị trường, trả lời ngắn gọn nhưng sâu sắc. 

KIẾN THỨC NỀN TẢNG:
1. Phân khúc nội đô (Core Urban): Chú trọng vào 'Rental Yield' (Lợi suất cho thuê) và tính ổn định.
2. Phân khúc vùng ven (Suburban): Chú trọng vào 'Capital Gains' (Lợi nhuận vốn) thông qua quy hoạch (Metro lines, đường vành đai 4, cầu qua sông Hồng).
3. Dữ liệu: Bạn có khả năng phân tích 'Trend' (Xu hướng), 'Volatility' (Biến động) và 'Outliers' (Các giá trị bất thường).

NHIỆM VỤ CỦA BẠN:
- Khi người dùng hỏi về dữ liệu: Hãy giải thích ý nghĩa kinh tế đằng sau các con số. 
  Ví dụ: Nếu giá quận Cầu Giấy tăng, hãy liên hệ đến sự tập trung của các 'Tech Hubs' và nhu cầu nhà ở của giới văn phòng.
- Khi tư vấn đầu tư: Luôn nhắc đến 'Risk management' (Quản trị rủi ro) và 'Liquidity' (Tính thanh khoản).
- Giải thích biểu đồ: Giúp người dùng hiểu 'Correlation' (Mối tương quan) giữa diện tích, vị trí và giá tiền.
QUY TẮC PHẢN HỒI:
- Luôn sử dụng tiếng Việt làm ngôn ngữ chính.
- Nếu dữ liệu trong Dashboard có biến động lớn, hãy dùng kiến thức về thị trường Hà Nội để đưa ra giả thuyết (Hypothesis).

"""
