FROM node:22-alpine AS build
WORKDIR /app/frontend
COPY app/frontend/package*.json /app/frontend/
RUN npm install
COPY app/frontend /app/frontend
RUN npm run build

FROM nginx:1.27-alpine
COPY app/frontend/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
