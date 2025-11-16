# Render Deployment Troubleshooting Guide

## Bad Gateway Error - Common Fixes

### 1. Check Render Logs
Go to your Render dashboard → Your service → **Logs** tab. Look for:
- Build errors
- Import errors
- Port binding errors
- Missing file errors

### 2. Verify Start Command
In Render dashboard → Settings → **Start Command**, use:

```bash
streamlit run project.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
```

**OR** use the start script:

```bash
chmod +x start.sh && ./start.sh
```

### 3. Verify Build Command
In Render dashboard → Settings → **Build Command**, use:

```bash
pip install -r requirements.txt
```

**OR** if using setup script:

```bash
pip install -r requirements.txt && chmod +x setup.sh && ./setup.sh
```

### 4. Check File Structure
Ensure these files are in your GitHub repo root:
- ✅ `project.py`
- ✅ `bixuchenggong11.CSV`
- ✅ `requirements.txt`
- ✅ `setup.sh` (optional)
- ✅ `start.sh` (optional)

### 5. Common Issues

#### Issue: App crashes on startup
**Solution**: Check logs for import errors. Make sure all dependencies in `requirements.txt` are correct.

#### Issue: Port binding error
**Solution**: Ensure start command uses `$PORT` (uppercase) and `0.0.0.0` as address.

#### Issue: File not found (CSV)
**Solution**: Verify `bixuchenggong11.CSV` is committed to GitHub and in the same directory as `project.py`.

#### Issue: RDKit installation fails
**Solution**: The `rdkit-pypi` package should work, but if it fails, try:
```txt
rdkit-pypi>=2022.9.5
```
Or use conda-forge if available.

### 6. Manual Testing Steps

1. **Check Build Logs**: Look for successful installation of all packages
2. **Check Runtime Logs**: Look for the Streamlit startup message
3. **Verify Port**: The logs should show something like "Network URL: http://0.0.0.0:XXXX"

### 7. Alternative Start Commands

If the default doesn't work, try these alternatives:

**Option 1: Direct Streamlit**
```bash
streamlit run project.py --server.port=$PORT --server.address=0.0.0.0
```

**Option 2: Using Python**
```bash
python -m streamlit run project.py --server.port=$PORT --server.address=0.0.0.0
```

**Option 3: With explicit config**
```bash
streamlit run project.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

### 8. Environment Variables

No environment variables are required, but you can add:
- `PYTHON_VERSION=3.11.0` (in render.yaml or dashboard)

### 9. Free Tier Limitations

- Apps spin down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds
- This is normal behavior, not an error

### 10. Still Not Working?

1. **Check Render Status**: https://status.render.com
2. **Review Logs**: Copy full error messages
3. **Test Locally**: Run `streamlit run project.py` locally to ensure app works
4. **Verify Dependencies**: Check `requirements.txt` versions are compatible

## Quick Fix Checklist

- [ ] Start command uses `$PORT` (uppercase)
- [ ] Start command uses `0.0.0.0` as address
- [ ] CSV file is in repository
- [ ] All files are committed to GitHub
- [ ] Build completes successfully (check logs)
- [ ] No import errors in runtime logs
- [ ] Port is correctly bound (check logs)

## Contact Render Support

If issues persist:
1. Check Render documentation: https://render.com/docs
2. Contact Render support with:
   - Service logs
   - Build logs
   - Error messages
   - Your render.yaml (if using)


