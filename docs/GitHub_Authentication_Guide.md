# GitHub Authentication Setup Guide

This guide explains how to set up SSH authentication for GitHub, which is especially important when multi-factor authentication (MFA) is enabled.


## Setup Steps

### 1. Generate an SSH Key

If you don't already have an SSH key, generate one:

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

**Notes:**
- Press Enter to accept the default location (`~/.ssh/id_ed25519`)
- Optionally set a passphrase for additional security
- Replace `your.email@example.com` with your actual email address

### 2. Copy Your Public SSH Key

Display and copy your public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

The output will look something like:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJl3dIeudNqd0DPMRD6OIh65tjkxFNOtwGcWB2gCgPhk your.email@example.com
```

### 3. Add the Key to GitHub

1. Go to GitHub → **Settings** → [**SSH and GPG keys**](https://github.com/settings/keys)
2. Click **"New SSH key"**
3. Give it a descriptive title (e.g., "Development Container" or "Work Laptop")
4. Paste your public key in the "Key" field
5. Click **"Add SSH key"**

### 4. Test the Connection

Verify that your SSH key is working:

```bash
ssh -T git@github.com
```

You should see a message like:
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

### 5. Configure Git

Set your git credentials globally (if not already configured):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Using SSH URLs

When cloning repositories or setting remotes, use SSH URLs instead of HTTPS:

```bash
# SSH format (recommended)
git clone git@github.com:username/repository.git

# HTTPS format (not recommended for authenticated operations)
git clone https://github.com/username/repository.git
```

To change an existing repository from HTTPS to SSH:

```bash
git remote set-url origin git@github.com:username/repository.git
```

## Troubleshooting

### Permission denied (publickey)

If you see this error, ensure:
1. Your SSH key is added to GitHub
2. You're using the correct SSH key
3. The SSH agent is running: `eval "$(ssh-agent -s)"`
4. Your key is added to the agent: `ssh-add ~/.ssh/id_ed25519`

### Could not resolve hostname

Check your internet connection and verify you can reach github.com:
```bash
ping github.com
```

### Using Multiple SSH Keys

If you have multiple GitHub accounts or SSH keys, configure them in `~/.ssh/config`:

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_work
```
