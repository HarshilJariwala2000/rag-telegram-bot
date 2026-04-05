# Use the official n8n image as a base
FROM docker.n8n.io/n8nio/n8n:latest

# Set working directory
WORKDIR /home/node

# Copy workflow JSON files from your host into the container
COPY ./workflows/*.json /home/node/workflows/

# Use a startup command to import workflows before starting n8n
# 'import:workflow' adds them to the database
# 'update:workflow --active=true' enables them immediately
CMD ["/usr/local/bin/n8n", "import:workflow", "--separate", "--input=/home/node/workflows/"]