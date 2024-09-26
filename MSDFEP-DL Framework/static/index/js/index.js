document.addEventListener('DOMContentLoaded', function () {
    // 监测点击改变高亮
    $('.menu>li').click(function (event) {
        $('.menu>li').removeClass('menu_select'); // 先移除所有菜单项的高亮
        $(this).addClass('menu_select'); // 然后只为当前点击的项添加高亮
    });

    // 监测点击 刷新.body_container中的内容
    $('.menu li').click(function () {
        // 获取被点击菜单项的id
        var menuItemId = $(this).attr('id');
        var url; // 定义 url 变量
        if (menuItemId == 'login') {
            url = './';
            window.location.href = url
        } else {
            url = './index/' + menuItemId;
            fetchContent(url);
            // console.log(url)
        }
    });

    // 定义加载内容的函数
    function fetchContent(url) {
        $.ajax({
            url: url,
            type: 'GET',
            success: function (response) {
                // 将获取的内容加载到.body_container中
                $('.body_container').html(response);
            },
            error: function () {
                $('.body_container').html('<p>内容加载失败，请重试。</p>');
            }
        });
    }
    // 自动加载“我的模型”内容
    fetchContent('./index/model_img_test');
    // 高亮“我的模型”菜单项
    $('#model_img_test').addClass('menu_select');

});