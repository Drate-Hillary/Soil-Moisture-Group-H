from django.shortcuts import redirect
from django.contrib.auth.decorators import user_passes_test

def role_required(role):
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            if request.user.role != role:
                return redirect('home')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def roles_required(*roles):
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect('login')
            if request.user.role not in roles:
                return redirect('home')
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator